"""Build the four-entity SQLite database."""
from pathlib import Path
import sqlite3
import pandas as pd
from .analysis import ACTIVITIES


def build_database(d: pd.DataFrame, output: Path, schema_path: Path) -> dict:
    frames={
        "Member":d[["Id","Age","Gender"]].rename(columns={"Id":"Member_ID"}),
        "Membership":d[["Id","EnrollmentStart","EnrollmentFinish","EnrollmentDuration",
                         "AverageFrequency","UseByTime","LifetimeValue"]].rename(columns={"Id":"Member_ID"}),
        "MembershipStatus":d[["Id","Dropout","NumberOfRenewals","AllowedNumberOfVisitsBySLA",
                               "RealNumberOfVisits","Ratio"]].rename(columns={"Id":"Member_ID"}),
        "Activity":pd.concat([pd.DataFrame({"Activity_Name":a,"Member_ID":d.loc[d[a].eq(1),"Id"]})
                               for a in ACTIVITIES],ignore_index=True),
    }
    output.mkdir(parents=True,exist_ok=True)
    target=output/"sports_centre.sqlite"
    tmp=output/"sports_centre.building.sqlite"
    if tmp.exists():tmp.unlink()
    with sqlite3.connect(tmp) as con:
        con.executescript(schema_path.read_text(encoding="utf-8"))
        for name,frame in frames.items():
            f=frame.copy()
            for c in f.select_dtypes(include=["datetime64[ns]"]).columns:
                f[c]=f[c].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            f.to_sql(name,con,if_exists="append",index=False)
        counts={name:con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in frames}
        fks=con.execute("PRAGMA foreign_key_check").fetchall()
        integrity=con.execute("PRAGMA integrity_check").fetchone()[0]
        summary=con.execute("SELECT COUNT(*), SUM(LifetimeValue) FROM MemberSummary").fetchone()
        if fks or integrity!="ok" or summary[0]!=len(d):
            raise ValueError(f"Database integrity failure: {fks}, {integrity}, {summary}")
        if abs(summary[1]-d.LifetimeValue.sum())>1e-6:
            raise ValueError("Customer-level payments were duplicated or lost")
        for key in ("Member","Membership","MembershipStatus"):
            if counts[key]!=len(d):raise ValueError(f"Missing customer summaries in {key}")
        if counts["Activity"]!=int(d.NumberOfActivities.sum()):
            raise ValueError("Activity membership rows disagree with source indicators")
    tmp.replace(target)
    return {"tables":counts,"foreign_key_violations":len(fks),"integrity_check":integrity,
            "member_summary_rows":summary[0],"historical_payments_total":summary[1]}
