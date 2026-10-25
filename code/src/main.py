# src/main.py

import json
import pathlib
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pdfminer.high_level import extract_text


# Configuration - root directory path relative to repository

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# Helper data structures

@dataclass
class DeletionRequest:
    case_id: str
    email: str
    request_timestamp: pd.Timestamp
    request_text: str

@dataclass
class DeletionRule:
    category: str
    action: str  # Edit / Delete / Retain / Anonymize

@dataclass
class SystemRecord:
    system_id: str
    resource_id: str
    category: str
    timestamp: pd.Timestamp
    state: str  # Edited / Deleted / Present / Anonymized

@dataclass
class Step:
    step_no: int
    request_id: str
    system_id: str
    resource_id: str
    action: str
    required_action: str
    evidence_id: Optional[str]
    step_json: str


# Load CSV Assets
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=True, keep_default_na=False)
    df.columns = df.columns.str.strip()
    return df


def load_all_assets():
    deletion_requests = load_csv("deletion_requests")
    user_identities = load_csv("user_identities")
    user_audit_log = load_csv("user_audit_log")
    system_activity = load_csv("system_activity")
    deletion_rules = load_csv("deletion_rules")
    evidence_files = load_csv("evidence_files")

    return {
        "deletion_requests": deletion_requests,
        "user_identities": user_identities,
        "user_audit_log": user_audit_log,
        "system_activity": system_activity,
        "deletion_rules": deletion_rules,
        "evidence_files": evidence_files,
    }


def resolve_identities(
    requests: pd.DataFrame,
    identities: pd.DataFrame,
    audit_log: pd.DataFrame,
) -> Dict[str, Set[str]]:

    # Mapping : case_id -> {current_user_id, historic_user_ids}
    # email -> current_user_id (most recent by updated_at)
    # user_identities.csv columns: user_id, name, email, phone, email_hash, updated_at
    identities = identities.copy()
    identities["updated_at"] = pd.to_datetime(identities["updated_at"])
    email_to_user = (
        identities.sort_values("updated_at")
        .groupby("email")
        .last()
        .reset_index()[["email", "user_id"]]
        .set_index("email")["user_id"]
        .to_dict()
    )

    # historic_id per user_id from audit_log
    # user_audit_log.csv columns: audit_id, user_id, event_time, field_name, old_value, new_value
    # A row with field_name == "user_id" records an identity merge/rename:
    # old_value = previous user_id, new_value = current user_id.
    audit_sorted = audit_log.sort_values("event_time")
    historic_map: Dict[str, Set[str]] = defaultdict(set)
    for _, row in audit_sorted.iterrows():
        if row.get("field_name") != "user_id":
            continue
        cur = row["new_value"]
        prev = row["old_value"]
        if pd.notnull(prev) and prev != "":
            historic_map[cur].add(prev)
            # chain backwards: include anything prev itself inherited
            historic_map[cur].update(historic_map.get(prev, set()))

    # Assemble per-case set
    case_to_ids: Dict[str, Set[str]] = {}
    for _, req in requests.iterrows():
        case = req["case_id"]
        email = req["email"]
        current = email_to_user.get(email)
        ids = {current} if current else set()
        if current and current in historic_map:
            ids.update(historic_map[current])
        case_to_ids[case] = {i for i in ids if pd.notnull(i) and i != ""}

    return case_to_ids

# Gather Evidence (CSV, JSON, SQLite, PDF)

def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    return extract_text(str(pdf_path))

def load_sqlite_snapshot(sql_path: pathlib.Path, user_ids: Set[str]) -> List[SystemRecord]:
    records = []
    conn = sqlite3.connect(sql_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    placeholder = ",".join("?" for _ in user_ids)
    query = f"""
        SELECT system_id, resource_id, category, timestamp, state
        FROM user_data
        WHERE user_id IN ({placeholder});
    """
    for row in cur.execute(query, tuple(user_ids)):
        records.append(
            SystemRecord(
                system_id=row["system_id"],
                resource_id=row["resource_id"],
                category=row["category"],
                timestamp=pd.to_datetime(row["timestamp"]),
                state=row["state"],
            )
        )
    conn.close()
    return records

def gather_system_records(
    case_id: str,
    user_ids: Set[str],
    evidence_files: pd.DataFrame,
) -> List[SystemRecord]:
    # evidence_files.csv columns: file_id, file_path, origin, captured_at
    records: List[SystemRecord] = []
    for _, f in evidence_files.iterrows():
        file_id = f["file_id"]
        rel_path = f["file_path"]
        file_type = pathlib.Path(rel_path).suffix.lstrip(".").lower()
        abs_path = DATA_DIR / rel_path

        if file_type == "csv":
            df = pd.read_csv(abs_path, parse_dates=True, keep_default_na=False)
            for _, row in df.iterrows():
                if row.get("user_id") in user_ids:
                    records.append(
                        SystemRecord(
                            system_id=row["system_id"],
                            resource_id=row["resource_id"],
                            category=row["category"],
                            timestamp=pd.to_datetime(row["timestamp"]),
                            state=row["state"],
                        )
                    )

        elif file_type == "json":
            with open(abs_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            
            system_id = data.get("source", file_id)
            category = data.get("collection", "unknown")
            timestamp = pd.to_datetime(data.get("captured_at"))
            for entry in data.get("records", []):
                uid = entry.get("user_id")
                if uid and uid in user_ids:
                    records.append(
                        SystemRecord(
                            system_id=system_id,
                            resource_id=entry["resource_id"],
                            category=category,
                            timestamp=timestamp,
                            state="PRESENT",
                        )
                    )

        elif file_type == "sqlite":
            records.extend(load_sqlite_snapshot(abs_path, user_ids))

        elif file_type == "pdf":
            text = extract_pdf_text(abs_path)
            for uid in user_ids:
                if re.search(rf"\b{re.escape(uid)}\b", text):
                    records.append(
                        SystemRecord(
                            system_id=file_id,
                            resource_id="PDF-" + file_id,
                            category="unstructured",
                            timestamp=pd.NaT,
                            state="PRESENT",
                        )
                    )

    return records


# Apply deletion rules
def build_rules_map(rules_df: pd.DataFrame) -> Dict[str, str]:
    # deletion_rules.csv columns: rule_id, data_category, condition, required_action
    return dict(zip(rules_df["data_category"], rules_df["required_action"]))

def tag_required_actions(
    records: List[SystemRecord], rules_map: Dict[str, str]
) -> List[Tuple[SystemRecord, str]]:
    tagged = []
    for rec in records:
        required = rules_map.get(rec.category, "RETAIN")  # default fallback
        tagged.append((rec, required.upper()))
    return tagged


# detect active flows / restore risks

def detect_active_flows(
    case_timestamp: pd.Timestamp,
    system_activity: pd.DataFrame,
    user_ids: Set[str],
) -> Tuple[Set[str], Set[str]]:
    if not user_ids:
        return set(), set()

    # system_activity.csv columns: activity_id, event_time, event_type, source_system,
    # target_system, resource_id, reported_result, details

    event_time = pd.to_datetime(system_activity["event_time"])
    later = system_activity[event_time > case_timestamp]

    # rows that mention any user_ids inside 'details'
    pattern = "|".join(re.escape(uid) for uid in user_ids)
    mask = later.apply(
        lambda r: bool(re.search(pattern, str(r.get("details")))),
        axis=1,
    )
    relevant = later[mask]

    # Classify - must filter on event_type (sync/export/backup/restore),
    # not event_time (a timestamp column, never equal to these strings).
    active_flows = set(relevant[relevant["event_type"].isin(["sync", "export"])]["activity_id"])
    restore_risks = set(
        relevant[relevant["event_type"].isin(["backup", "restore"])]["activity_id"]
    )
    return active_flows, restore_risks


# Detect Overall Detection status
def compute_status(
    tagged_records: List[Tuple[SystemRecord, str]],
    active_flows: Set[str],
    restore_risks: Set[str],
    request_text: str,
) -> str:

    # Returns one of: INCOMPLETE, COMPLETE, NOT_APPLICABLE
    intent = bool(re.search(r"\b(delete|remove|erase|forget|right to be forgotten)\b", request_text, re.I))

    if not intent:
        return "NOT_APPLICABLE"

    for rec, required in tagged_records:
        if required in ("DELETE", "EDIT", "ANONYMIZE"):
            if rec.state == "PRESENT":
                return "INCOMPLETE"

    if active_flows or restore_risks:
        return "INCOMPLETE"

    return "COMPLETE"


# Generate Ordered Steps in JSON
def generate_steps(
    case_id: str,
    tagged_records: List[Tuple[SystemRecord, str]],
    active_flows: Set[str],
    restore_risks: Set[str],
    evidence_files: pd.DataFrame,
) -> List[Step]:
    steps: List[Step] = []
    step_no = 1

    evidence_id_by_record: Dict[Tuple[str, str], str] = {}
    sorted_records = sorted(tagged_records, key=lambda x: (x[0].system_id, x[0].resource_id))

    # Phase 1: FINDING
    for rec, required in sorted_records:
        evidence_id = evidence_id_by_record.get((rec.system_id, rec.resource_id))
        step = Step(
            step_no=step_no,
            request_id=case_id,
            system_id=rec.system_id,
            resource_id=rec.resource_id,
            action="FINDING",
            required_action=required,
            evidence_id=evidence_id,
            step_json=json.dumps(
                {
                    "record": asdict(rec),
                    "required_action": required,
                },
                default=str,
            ),
        )
        steps.append(step)
        step_no += 1

    # Phase 2: REMEDIATION
    needed_actions = {req for _, req in tagged_records if req != "RETAIN"}
    for action in sorted(needed_actions):
        step = Step(
            step_no=step_no,
            request_id=case_id,
            system_id=None,  # Not applicable
            resource_id=None,  # Not applicable
            action="REMEDIATION",
            required_action=action,
            evidence_id=None,  # Not applicable
            step_json=json.dumps({"remediate_action": action}),
        )
        steps.append(step)
        step_no += 1

    # Phase 3: VERIFICATION of records
    for rec, required in sorted_records:
        evidence_id = evidence_id_by_record.get((rec.system_id, rec.resource_id))
        step = Step(
            step_no=step_no,
            request_id=case_id,
            system_id=rec.system_id,
            resource_id=rec.resource_id,
            action="VERIFICATION",
            required_action=required,
            evidence_id=evidence_id,
            step_json=json.dumps(
                {
                    "record": asdict(rec),
                    "verified_state": rec.state,
                },
                default=str,
            ),
        )
        steps.append(step)
        step_no += 1

    # Phase 3b: VERIFICATION of active flows
    for act_id in sorted(active_flows):
        steps.append(
            Step(
                step_no=step_no,
                request_id=case_id,
                system_id=None,
                resource_id=None,
                action="VERIFICATION",
                required_action="ACTIVE_FLOW",
                evidence_id=act_id,
                step_json=json.dumps({"type": "sync_or_export", "activity_id": act_id}),
            )
        )
        step_no += 1

    # Phase 3c: VERIFICATION of restore risks
    for act_id in sorted(restore_risks):
        steps.append(
            Step(
                step_no=step_no,
                request_id=case_id,
                system_id=None,
                resource_id=None,
                action="VERIFICATION",
                required_action="RESTORE_RISK",
                evidence_id=act_id,
                step_json=json.dumps({"type": "backup_or_restore", "activity_id": act_id}),
            )
        )
        step_no += 1

    return steps


# CSV Output
def write_output_csv(all_steps: List[Step]):
    rows = [
        {
            "step_no": s.step_no,
            "request_id": s.request_id,
            "system_id": s.system_id,
            "resource_id": s.resource_id,
            "action": s.action,
            "required_action": s.required_action,
            "evidence_id": s.evidence_id,
            "step_json": s.step_json,
        }
        for s in all_steps
    ]
    out_df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "output.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Output written to {out_path}")


# Main
def main():
    assets = load_all_assets()

    # Pre-compute rule look-up
    rules_map = build_rules_map(assets["deletion_rules"])

    # Resolve identities for every request
    case_to_userids = resolve_identities(
        assets["deletion_requests"],
        assets["user_identities"],
        assets["user_audit_log"],
    )

    all_steps: List[Step] = []

    for _, req_row in assets["deletion_requests"].iterrows():
        case_id = req_row["case_id"]
        request_ts = pd.to_datetime(req_row["requested_at"])
        request_txt = req_row["request_text"]

        user_ids = case_to_userids.get(case_id, set())

        # Defaults in case there is no resolved user (so generate_steps always
        # has something valid to work with).
        tagged: List[Tuple[SystemRecord, str]] = []
        active_flows: Set[str] = set()
        restore_risks: Set[str] = set()

        if not user_ids:
            # Status: No user found
            status = "NOT_APPLICABLE"
        else:
            # Gather evidence files
            records = gather_system_records(case_id, user_ids, assets["evidence_files"])

            # Apply rules
            tagged = tag_required_actions(records, rules_map)

            # Detect active flow / restore risks
            active_flows, restore_risks = detect_active_flows(
                request_ts, assets["system_activity"], user_ids
            )

            # Compute status
            status = compute_status(tagged, active_flows, restore_risks, request_txt)

        # Generate steps irrespective of status
        steps = generate_steps(
            case_id,
            tagged,
            active_flows,
            restore_risks,
            assets["evidence_files"],
        )
        all_steps.extend(steps)

        print(f"Processed case {case_id}: status={status}")

    # Creates CSV output file
    write_output_csv(all_steps)


if __name__ == "__main__":
    main()