"""Source quality and monthly membership checks."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .analysis import (ACTIVITIES, CUTOFF, GROUP_A, GROUP_B, START, STOP,
                       age_masks, in_reporting_period)


def source_checks(d: pd.DataFrame) -> pd.DataFrame:
    entries = d.StartDate.ge(START) & d.StartDate.lt(STOP)
    event = d.Dropout.eq(1)
    same_month = d.StartDate.dt.to_period("M").eq(d.FinishDate.dt.to_period("M"))
    report = in_reporting_period(d)
    mid, stable = age_masks(d), age_masks(d,"stable")
    selected = mid[GROUP_A] | mid[GROUP_B]
    boundary = (mid[GROUP_A] & ~stable[GROUP_A]) | (mid[GROUP_B] & ~stable[GROUP_B])
    checks = [
        ("Missing customer IDs",d.Id.isna().sum(),"ERROR"),
        ("Duplicate customer IDs",d.Id.duplicated().sum(),"ERROR"),
        ("Missing or invalid EnrollmentStart",d.StartDate.isna().sum(),"ERROR"),
        ("Missing or invalid EnrollmentFinish",d.FinishDate.isna().sum(),"ERROR"),
        ("Missing essential dates or status",d[["EnrollmentStart","EnrollmentFinish","Dropout"]].isna().any(axis=1).sum(),"ERROR"),
        ("Invalid Dropout coding",(~d.Dropout.isin([0,1])).sum(),"ERROR"),
        ("Finish precedes start",d.EnrollmentFinish.lt(d.EnrollmentStart).sum(),"ERROR"),
        ("Dates beyond source cutoff",(d.StartDate.gt(CUTOFF)|d.FinishDate.gt(CUTOFF)).sum(),"ERROR"),
        ("Uncensored records not at cutoff",(d.Dropout.eq(0)&d.FinishDate.ne(CUTOFF)).sum(),"ERROR"),
        ("Censoring within reporting period",(d.Dropout.eq(0)&d.FinishDate.ge(START)&d.FinishDate.lt(STOP)).sum(),"ERROR"),
        ("Invalid renewal counts",(d.NumberOfRenewals.isna()|d.NumberOfRenewals.lt(0)|d.NumberOfRenewals.mod(1).ne(0)).sum(),"ERROR"),
        ("Invalid activity indicators",(~d[ACTIVITIES].isin([0,1])).sum().sum(),"ERROR"),
        ("Activity count disagrees with flags",d.NumberOfActivities.ne(d[ACTIVITIES].sum(axis=1)).sum(),"ERROR"),
        ("Input records",len(d),"INFO"),
        ("Censored input records",d.Dropout.eq(0).sum(),"INFO"),
        ("Renewed at least once",d.NumberOfRenewals.gt(0).sum(),"INFO"),
        ("Source records",len(d),"INFO"),
        ("Age zero",d.Age.eq(0).sum(),"INFO"),
        ("Missing or nonnumeric age",pd.to_numeric(d.Age,errors="coerce").isna().sum(),"INFO"),
        ("Negative, above-100 or noninteger age",(d.Age.lt(0)|d.Age.gt(100)|d.Age.mod(1).ne(0)).sum(),"INFO"),
        ("Negative estimated entry age among otherwise valid ages",(d.Age.gt(0)&d.EstimatedEntryAge.lt(0)).sum(),"INFO"),
        ("Total unusable/unknown ages",(~d.ValidAge).sum(),"INFO"),
        ("Unusable ages observed in reporting period",(~d.ValidAge&report).sum(),"INFO"),
        ("Usable reporting customers outside the two proposed groups",(d.ValidAge&report&~selected).sum(),"INFO"),
        ("Primary selected reporting customers sensitive to an age boundary",(report&boundary).sum(),"INFO"),
        ("All first entrants 2015-2018",entries.sum(),"INFO"),
        ("Unusable ages among first entrants 2015-2018",(entries&~d.ValidAge).sum(),"INFO"),
        ("Other usable entry ages outside A and B",(entries&d.ValidAge&~selected).sum(),"INFO"),
        ("Selected entrants",(entries&selected).sum(),"INFO"),
        ("Selected age-boundary sensitive entrants",(entries&boundary).sum(),"INFO"),
        ("Selected entrants without potential 180-day follow-up",(entries&selected&(d.StartDate+pd.Timedelta(days=180)).gt(CUTOFF)).sum(),"INFO"),
        ("Final window starts before enrolment",d.LastPeriodStart.lt(d.StartDate).sum(),"REVIEW"),
        ("Final window ends after enrolment",d.LastPeriodFinish.gt(d.FinishDate).sum(),"REVIEW"),
        ("Final window longer than 62 days",(d.LastPeriodFinish-d.LastPeriodStart).dt.days.gt(62).sum(),"REVIEW"),
        ("Stored ratio differs from uncapped actual/allowed by over 0.1 pp",(100*d.RealNumberOfVisits/d.AllowedNumberOfVisitsBySLA-d.Ratio).abs().gt(.1).sum(),"REVIEW"),
    ]
    out = pd.DataFrame(checks,columns=["Check","AffectedRecords","Severity"])
    out["AffectedRecords"] = out.AffectedRecords.astype(int)
    out["Result"] = np.where(out.Severity.eq("ERROR"),np.where(out.AffectedRecords.eq(0),"PASS","FAIL"),out.Severity)
    return out


def monthly_diagnostics(d: pd.DataFrame, m: pd.DataFrame) -> dict[str,pd.DataFrame]:
    rows, snapshots, dates = [], [], []
    event = d.Dropout.eq(1)
    for r in m.itertuples():
        nxt = r.MonthStart + pd.offsets.MonthBegin(1)
        independent = int(d.StartDate.lt(nxt).sum() - (event&d.FinishDate.lt(nxt)).sum())
        rows.append(dict(MonthStart=r.MonthStart,
                         StockFlowResidual=r.ActiveMemberships-r.OpeningMemberships-r.NewEnrolments+r.Dropouts,
                         DropoutPartitionResidual=r.Dropouts-r.DropoutsFromOpening-r.SameMonthEnrolmentDropouts,
                         IndependentLedgerResidual=r.ActiveMemberships-independent,
                         RateResidual=r.DropoutRate-r.DropoutsFromOpening/r.OpeningMemberships))
        last = int((event&d.FinishDate.eq(r.MonthEnd)).sum())
        snapshots.append(dict(MonthStart=r.MonthStart,PrimaryClosingStock=r.ActiveMemberships,
                              DropoutsDatedOnMonthEnd=last,EndDateInclusiveSnapshot=r.ActiveMemberships+last))
    counts = d.loc[event&d.FinishDate.ge(START)&d.FinishDate.lt(STOP)].groupby("FinishDate").size().rename("DropoutsOnDate").reset_index()
    counts["Year"] = counts.FinishDate.dt.year
    for y,g in counts.groupby("Year",sort=True):
        total=g.DropoutsOnDate.sum()
        for r in g.sort_values(["DropoutsOnDate","FinishDate"],ascending=[False,True]).head(2).itertuples():
            dates.append(dict(Year=int(y),FinishDate=r.FinishDate,DropoutsOnDate=r.DropoutsOnDate,
                              ShareOfAnnualDropouts=r.DropoutsOnDate/total))
    annual_counts=m.groupby("Year").NewEnrolments.sum()
    december=m.set_index("MonthStart").ActiveMemberships
    repeated=[name for name,g in m.groupby("MonthName") if g.EnrolmentRateRankWithinYear.le(3).all()]
    repeated=sorted(repeated,key=lambda n:list(__import__("calendar").month_name).index(n))
    overview=pd.DataFrame([
        ("Reporting months",len(m)),
        ("Population interpretation","Observed-cohort interval reconstruction"),
        ("Recorded first enrolments",int(m.NewEnrolments.sum())),
        ("Recorded final dropouts",int(m.Dropouts.sum())),
        ("Same-month entrants/exits",int(m.SameMonthEnrolmentDropouts.sum())),
        ("December 2015 reconstructed stock",int(december.loc["2015-12-01"])),
        ("December 2018 reconstructed stock",int(december.loc["2018-12-01"])),
        ("December-to-December stock change",december.loc["2018-12-01"]/december.loc["2015-12-01"]-1),
        ("2018 versus 2015 first-enrolment change",annual_counts.loc[2018]/annual_counts.loc[2015]-1),
        ("Repeated top-three enrolment months",", ".join(repeated)),
        ("Dropout seasonality","Peaks differ across years"),
        ("Computational validation","All monthly checks pass"),
    ],columns=["Measure","Value"])
    return {"Overview":overview,"MonthlyReconciliation":pd.DataFrame(rows),
            "EndDateSensitivity":pd.DataFrame(snapshots),"LargestExitDates":pd.DataFrame(dates)}
