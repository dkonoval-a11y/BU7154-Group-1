"""Membership preparation, seasonality and cohort retention calculations."""
from __future__ import annotations

import calendar
from collections import OrderedDict

import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp("2019-10-31")
START = pd.Timestamp("2015-01-01")
STOP = pd.Timestamp("2019-01-01")
MONTHS = pd.date_range(START, STOP, freq="MS", inclusive="left")
GROUP_A = "A: entry age 18-24"
GROUP_B = "B: entry age 35+"
GROUPS = (GROUP_A, GROUP_B)
ACTIVITIES = [f"{a}Activities" for a in
              ("Athletics", "Water", "Fitness", "Dance", "Team", "Racket",
               "Combat", "Nature", "Special", "Other")]
MASTER_COLUMNS = [
    "Id", "Age", "Gender", "EnrollmentStart", "EnrollmentFinish",
    "EnrollmentDuration", "EnrollmentDurationClass1", "EnrollmentDurationClass2",
    "FirstAnniversaryDate", "ObservedTenureDays", "LifetimeValue", "Dropout",
    "AverageFrequency", "AverageFrequencyClass1", "AverageFrequencyClass2",
    "Ratio", "RatioClass2", "AllowedWeeklyVisitsBySLA",
    "AllowedNumberOfVisitsBySLA", "RealNumberOfVisits", "LastPeriodStart",
    "LastPeriodFinish", "UseByTime", "NumberOfRenewals", *ACTIVITIES,
    "NumberOfActivities",
]


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select source fields and calculate the two date helpers."""
    d = raw.copy()
    for c in ("EnrollmentStart", "EnrollmentFinish", "LastPeriodStart", "LastPeriodFinish"):
        d[c] = pd.to_datetime(d[c], errors="raise")
    s = d.EnrollmentStart.dt.normalize()
    f = d.EnrollmentFinish.dt.normalize()
    d["FirstAnniversaryDate"] = s + pd.DateOffset(months=12)
    d["ObservedTenureDays"] = (f - s).dt.days
    master = d[MASTER_COLUMNS].copy()
    d["StartDate"] = s
    d["FinishDate"] = f
    d["AgeReferenceDate"] = f.where(d.Dropout.eq(1), CUTOFF)
    elapsed = (d.AgeReferenceDate - s).dt.days / 365.2425
    d["EntryAgeLower"] = d.Age - elapsed
    d["EstimatedEntryAge"] = d.EntryAgeLower + 0.5
    d["EntryAgeUpper"] = d.EntryAgeLower + 1.0
    d["ValidAge"] = (d.Age.between(1, 100) & d.Age.mod(1).eq(0)
                     & d.EstimatedEntryAge.ge(0))
    return master, d


def age_masks(d: pd.DataFrame, scenario: str = "midpoint") -> dict[str, pd.Series]:
    if scenario == "stable":
        a = d.EntryAgeLower.ge(18) & d.EntryAgeUpper.lt(25)
        b = d.EntryAgeLower.ge(35)
    else:
        name = {"midpoint": "EstimatedEntryAge", "lower": "EntryAgeLower",
                "upper": "EntryAgeUpper"}[scenario]
        a = d[name].ge(18) & d[name].lt(25)
        b = d[name].ge(35)
    return {GROUP_A: a & d.ValidAge, GROUP_B: b & d.ValidAge}


def in_reporting_period(d: pd.DataFrame) -> pd.Series:
    """Any overlap with reporting dates, including an exit on 1 January."""
    return d.StartDate.lt(STOP) & (~d.Dropout.eq(1) | d.FinishDate.ge(START))


def monthly(d: pd.DataFrame) -> pd.DataFrame:
    """Apply exits on their recorded dates; retain pre-2015 carry-in members."""
    rows = []
    event = d.Dropout.eq(1)
    for m in MONTHS:
        nxt = m + pd.offsets.MonthBegin(1)
        opening = d.StartDate.lt(m) & (~event | d.FinishDate.ge(m))
        new = d.StartDate.ge(m) & d.StartDate.lt(nxt)
        exit_ = event & d.FinishDate.ge(m) & d.FinishDate.lt(nxt)
        closing = d.StartDate.lt(nxt) & (~event | d.FinishDate.ge(nxt))
        op, n, x, cl = (int(z.sum()) for z in (opening, new, exit_, closing))
        xo = int((opening & exit_).sum())
        same = int((new & exit_).sum())
        rows.append(dict(
            MonthStart=m, MonthEnd=nxt - pd.Timedelta(days=1), Year=m.year,
            MonthNumber=m.month, MonthName=calendar.month_name[m.month],
            OpeningMemberships=op, NewEnrolments=n, Dropouts=x,
            ActiveMemberships=cl, DropoutsFromOpening=xo,
            SameMonthEnrolmentDropouts=same,
            EnrolmentRate=n / op if op else np.nan,
            DropoutRate=xo / op if op else np.nan,
            NetMembershipChange=cl - op,
            NetMembershipChangeRate=(cl - op) / op if op else np.nan))
    out = pd.DataFrame(rows)
    years = out.groupby("Year")
    out["AnnualEnrolmentShare"] = out.NewEnrolments / years.NewEnrolments.transform("sum")
    out["AnnualDropoutShare"] = out.Dropouts / years.Dropouts.transform("sum")
    out["ActiveIndexWithinYear"] = out.ActiveMemberships / years.ActiveMemberships.transform("mean")
    for c in ("EnrolmentRate", "DropoutRate"):
        out[c + "RankWithinYear"] = out.groupby("Year")[c].rank(ascending=False, method="min").astype("Int64")
    return out


def annual(m: pd.DataFrame, age_group: str | None = None) -> pd.DataFrame:
    rows = []
    for year, g in m.groupby("Year", sort=True):
        op = int(g.OpeningMemberships.iloc[0])
        cl = int(g.ActiveMemberships.iloc[-1])
        base = g.OpeningMemberships.sum()
        r = dict(Year=int(year), OpeningMemberships=op,
                 NewEnrolments=int(g.NewEnrolments.sum()), Dropouts=int(g.Dropouts.sum()),
                 ClosingMemberships=cl, NetMembershipChange=cl-op,
                 AnnualNetChangeRate=(cl-op)/op)
        if age_group is None:
            r["MeanActiveMemberships"] = g.ActiveMemberships.mean()
        r["MeanMonthlyEnrolmentRate"] = g.EnrolmentRate.mean()
        r["MeanMonthlyDropoutRate"] = g.DropoutRate.mean()
        prefix = "ExposureWeighted" if age_group is None else "Weighted"
        r[prefix + "MonthlyEnrolmentRate"] = g.NewEnrolments.sum()/base
        r[prefix + "MonthlyDropoutRate"] = g.DropoutsFromOpening.sum()/base
        if age_group is None:
            r["SameMonthEnrolmentDropouts"] = int(g.SameMonthEnrolmentDropouts.sum())
            r["Q3ShareOfAnnualDropouts"] = g.loc[g.MonthNumber.between(7,9), "Dropouts"].sum()/g.Dropouts.sum()
            r["PeakDropoutMonth"] = g.loc[g.Dropouts.idxmax(), "MonthName"]
        else:
            r = {"AgeGroup": age_group, **r}
        rows.append(r)
    return pd.DataFrame(rows)


def seasonal(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for num, g in m.groupby("MonthNumber", sort=True):
        r = dict(MonthNumber=int(num), MonthName=calendar.month_name[num], YearsObserved=g.Year.nunique())
        for c in ("NewEnrolments", "Dropouts", "ActiveMemberships", "NetMembershipChange"):
            r["Mean" + c] = g[c].mean()
        for c in ("EnrolmentRate", "DropoutRate"):
            for prefix, method in (("Mean","mean"),("Median","median"),("Min","min"),("Max","max")):
                r[prefix+c] = getattr(g[c], method)()
        r["PooledEnrolmentRate"] = g.NewEnrolments.sum()/g.OpeningMemberships.sum()
        r["PooledDropoutRate"] = g.DropoutsFromOpening.sum()/g.OpeningMemberships.sum()
        for c in ("AnnualEnrolmentShare", "AnnualDropoutShare", "ActiveIndexWithinYear"):
            r["Mean"+c] = g[c].mean()
        r["MeanNetChangeRate"] = g.NetMembershipChangeRate.mean()
        for c in ("EnrolmentRate", "DropoutRate"):
            r["YearsTop3"+c] = int(g[c+"RankWithinYear"].le(3).sum())
        r["YearsWithNetGrowth"] = int(g.NetMembershipChange.gt(0).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def paired_months(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    keys = ["MonthStart", "Year", "MonthNumber", "MonthName"]
    p = a.merge(b, on=keys, suffixes=("A", "B"), validate="one_to_one")
    r = p[keys].copy()
    for metric, out in (("EnrolmentRate","Enrolment"), ("DropoutRate","Dropout")):
        r[metric+"A"] = p[metric+"A"]
        r[metric+"B"] = p[metric+"B"]
        r[out+"GapPP"] = 100*(p[metric+"A"]-p[metric+"B"])
    for c in ("A", "B"):
        r["AnnualEnrolmentShare"+c] = p["AnnualEnrolmentShare"+c]
    return r


def age_seasonal(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for num, g in p.groupby("MonthNumber", sort=True):
        r = dict(MonthNumber=int(num), MonthName=calendar.month_name[num])
        for metric, short in (("EnrolmentRate", "Enrolment"), ("DropoutRate", "Dropout")):
            r["Mean"+metric+"A"] = g[metric+"A"].mean()
            r["Mean"+metric+"B"] = g[metric+"B"].mean()
            gap = g[short+"GapPP"]
            r["Mean"+short+"GapPP"] = gap.mean()
            r["YearsAHigher"+short] = int(gap.gt(0).sum())
            r["Minimum"+short+"GapPP"] = gap.min()
            r["Maximum"+short+"GapPP"] = gap.max()
        for suffix in ("A", "B"):
            r["MeanAnnualEnrolmentShare"+suffix] = g["AnnualEnrolmentShare"+suffix].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def focus_windows(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group, frame in ((GROUP_A, a), (GROUP_B, b)):
        for year, g in frame.groupby("Year", sort=True):
            for label, months in (("February-March", [2,3]), ("July-September", [7,8,9])):
                sub = g[g.MonthNumber.isin(months)]
                base = int(sub.OpeningMemberships.sum())
                rows.append(dict(AgeGroup=group, Year=int(year), Window=label,
                                 NewEnrolments=int(sub.NewEnrolments.sum()), Dropouts=int(sub.Dropouts.sum()),
                                 OpeningMemberMonths=base,
                                 WeightedMonthlyEnrolmentRate=sub.NewEnrolments.sum()/base,
                                 WeightedMonthlyDropoutRate=sub.DropoutsFromOpening.sum()/base))
    f = pd.DataFrame(rows)
    p = f[f.AgeGroup.eq(GROUP_A)].merge(f[f.AgeGroup.eq(GROUP_B)], on=["Year","Window"], suffixes=("A","B"))
    gap = p[["Year","Window"]].copy()
    for short in ("Enrolment", "Dropout"):
        gap[short+"GapPP"] = 100*(p["WeightedMonthly"+short+"RateA"]-p["WeightedMonthly"+short+"RateB"])
    return f, gap


def group_counts(d: pd.DataFrame) -> pd.DataFrame:
    mid = age_masks(d)
    stable = age_masks(d, "stable")
    report = in_reporting_period(d)
    entries = d.StartDate.ge(START) & d.StartDate.lt(STOP)
    exits = d.Dropout.eq(1) & d.FinishDate.ge(START) & d.FinishDate.lt(STOP)
    data = OrderedDict()
    for group in GROUPS:
        mask = mid[group]
        m = monthly(d.loc[mask])
        data[group] = OrderedDict(
            AllSourceHistoryRecords=int(mask.sum()), ReportingPeriodCustomers=int((mask & report).sum()),
            FirstEnrolments2015_2018=int((mask & entries).sum()),
            RecordedDropouts2015_2018=int((mask & exits).sum()),
            OpeningJanuary2015=int(m.OpeningMemberships.iloc[0]),
            ClosingDecember2018=int(m.ActiveMemberships.iloc[-1]),
            MinimumMonthlyOpeningBase=int(m.OpeningMemberships.min()),
            MaximumMonthlyOpeningBase=int(m.OpeningMemberships.max()),
            BoundarySensitiveReportingCustomers=int((mask & ~stable[group] & report).sum()),
            SameMonthEntrantExits=int(m.SameMonthEnrolmentDropouts.sum()))
    return pd.DataFrame({"Measure": list(data[GROUP_A]), **{g:list(v.values()) for g,v in data.items()}})


def age_results(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = age_masks(d)
    ms = {g:monthly(d.loc[mask]) for g,mask in masks.items()}
    combined = pd.concat([m.assign(AgeGroup=g) for g,m in ms.items()], ignore_index=True)
    age_cols = ["AgeGroup", *[c for c in ms[GROUP_A].columns if c not in
                ("ActiveIndexWithinYear","EnrolmentRateRankWithinYear","DropoutRateRankWithinYear")]]
    paired = paired_months(ms[GROUP_A], ms[GROUP_B])
    focus, fgaps = focus_windows(ms[GROUP_A], ms[GROUP_B])
    sens, sens_season = [], []
    for name, mode in (("Midpoint estimate","midpoint"), ("Lower-age bound","lower"),
                       ("Upper-age bound","upper"), ("Stable group membership only","stable")):
        scenario_months = {}
        for group, mask in age_masks(d, mode).items():
            m = monthly(d.loc[mask]); scenario_months[group] = m
            sens.append(dict(Scenario=name, AgeGroup=group,
                             ReportingPeriodCustomers=int((mask & in_reporting_period(d)).sum()),
                             MinimumOpeningBase=int(m.OpeningMemberships.min()),
                             MeanEnrolmentRate=m.EnrolmentRate.mean(), MeanDropoutRate=m.DropoutRate.mean()))
        ss = age_seasonal(paired_months(scenario_months[GROUP_A], scenario_months[GROUP_B]))
        ss.insert(0, "Scenario", name); sens_season.append(ss)
    return {
        "GroupsAndChecks": group_counts(d), "MonthlyData": combined[age_cols],
        "AnnualSummary": pd.concat([annual(m,g) for g,m in ms.items()], ignore_index=True),
        "SeasonalComparison": age_seasonal(paired), "YearlyRateGaps": paired,
        "FocusWindows": focus, "FocusWindowGaps": fgaps,
        "Sensitivity": pd.DataFrame(sens),
        "SensitivitySeasonal": pd.concat(sens_season, ignore_index=True),
    }


def retention_cohorts(d: pd.DataFrame, mode: str = "midpoint", inclusive_final_day: bool = False
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No final exit by horizon; early known exits remain eligible failures."""
    entrants = d.StartDate.ge(START) & d.StartDate.lt(STOP)
    masks = age_masks(d, mode)
    rows = []
    for h in (90,180):
        for month in MONTHS:
            nxt = month + pd.offsets.MonthBegin(1)
            for group in GROUPS:
                q = d.loc[entrants & masks[group] & d.StartDate.ge(month) & d.StartDate.lt(nxt)]
                target = q.StartDate + pd.Timedelta(days=h)
                failure = q.Dropout.eq(1) & (q.FinishDate.lt(target) if inclusive_final_day else q.FinishDate.le(target))
                eligible = failure | q.FinishDate.ge(target)
                n, kept = int(eligible.sum()), int((eligible & ~failure).sum())
                rows.append(dict(CohortMonth=month, AgeGroup=group, HorizonDays=h,
                                 CohortSize=len(q), EligibleN=n, RetainedN=kept,
                                 ExitsByHorizon=int(failure.sum()), CensoredBeforeHorizon=int((~eligible).sum()),
                                 RetentionRate=kept/n if n else np.nan, CohortYear=month.year,
                                 MonthNumber=month.month, MonthName=calendar.month_name[month.month],
                                 SmallCohort=n<20))
    c = pd.DataFrame(rows)
    keys = ["CohortMonth","CohortYear","MonthNumber","MonthName","HorizonDays"]
    p = c[c.AgeGroup.eq(GROUP_A)].merge(c[c.AgeGroup.eq(GROUP_B)], on=keys, suffixes=("A","B"), validate="one_to_one")
    comp = p[keys].copy()
    for group in ("A","B"):
        for metric in ("EligibleN","RetainedN","RetentionRate"):
            comp[metric+group] = p[metric+group]
    comp["GapPP_AminusB"] = 100*(comp.RetentionRateA-comp.RetentionRateB)
    comp["PairedSupport"] = comp.EligibleNA.gt(0) & comp.EligibleNB.gt(0)
    comp["SmallEitherGroup"] = comp.EligibleNA.lt(20) | comp.EligibleNB.lt(20)
    comp["PooledCohortWeightN"] = comp.EligibleNA + comp.EligibleNB
    return c, comp


def retention_summary(p: pd.DataFrame, label: str, equal_weights: bool = False) -> dict:
    """Raw rates over selected cohorts; adjusted rates over common support."""
    if p.HorizonDays.nunique() != 1:
        raise ValueError("Select exactly one horizon before summarising retention")
    support = p.loc[p.PairedSupport]
    w = pd.Series(1.0,index=support.index) if equal_weights else support.PooledCohortWeightN
    gap = support.GapPP_AminusB
    r = dict(Scope=label, HorizonDays=int(p.HorizonDays.iloc[0]),
             EligibleNA=int(p.EligibleNA.sum()), EligibleNB=int(p.EligibleNB.sum()))
    for group in ("A","B"):
        denom = p["EligibleN"+group].sum()
        r["RawRate"+group] = p["RetainedN"+group].sum()/denom if denom else np.nan
    for group in ("A","B"):
        r["MatchedRate"+group] = (support["RetentionRate"+group]*w).sum()/w.sum() if w.sum() else np.nan
    r["GapPP_AminusB"] = 100*(r["MatchedRateA"]-r["MatchedRateB"])
    r["PairedCohorts"] = len(support)
    r["BCohortsHigher"] = int(gap.lt(-1e-12).sum())
    r["ACohortsHigher"] = int(gap.gt(1e-12).sum())
    r["TiedCohorts"] = int(gap.abs().le(1e-12).sum())
    r["SparseCohorts"] = int(support.SmallEitherGroup.sum())
    return r


def retention_results(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    c, p = retention_cohorts(d)
    summaries = []
    for h in (90,180):
        q = p[p.HorizonDays.eq(h)]
        scopes = [("All 2015-2018",q)] + [(str(y),q[q.CohortYear.eq(y)]) for y in range(2015,2019)]
        scopes += [(calendar.month_name[m],q[q.MonthNumber.eq(m)]) for m in (3,9,10)]
        scopes += [("Other nine months",q[~q.MonthNumber.isin([3,9,10])])]
        summaries.extend(retention_summary(frame,label) for label,frame in scopes)
    sensitivity = []
    for label,mode in (("Lower entry-age bound","lower"),("Upper entry-age bound","upper"),
                       ("Stable age assignments only","stable"),("Final day included","midpoint")):
        _, q = retention_cohorts(d, mode, inclusive_final_day=label=="Final day included")
        for h in (90,180):
            sensitivity.append(retention_summary(q[q.HorizonDays.eq(h)],label))
    for h in (90,180):
        q = p[p.HorizonDays.eq(h)]
        sensitivity.append(retention_summary(q[~q.SmallEitherGroup], "Both groups N >= 20"))
        sensitivity.append(retention_summary(q, "Equal weight per exact cohort", equal_weights=True))
    return {"Summary":pd.DataFrame(summaries), "CohortRetention":c,
            "CohortComparisons":p, "Sensitivity":pd.DataFrame(sensitivity)}


def descriptive_results(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarise recorded-age activities, attendance and renewals."""
    q = d.copy()
    q["ActivityAgeGroup"] = pd.cut(q.Age, [-np.inf,17,24,34,44,54,64,np.inf],
                                    labels=["Under 18","18-24","25-34","35-44","45-54","55-64","65+"])
    heat = q.groupby(["ActivityAgeGroup","Gender"],observed=True).NumberOfActivities.agg(["size","mean"]).reset_index()
    heat.columns = ["ActivityAgeGroup","Gender","Members","AverageActivityVariety"]
    rows = []
    for label,mask in (("18-24",q.Age.between(18,24)),("25+",q.Age.ge(25))):
        for activity in ACTIVITIES:
            rows.append(dict(AgeGroup=label, Activity=activity, MembersWithActivity=int(q.loc[mask,activity].sum()),
                             MembersInAgeGroup=int(mask.sum())))
    renew = q.groupby("NumberOfRenewals").size().rename("Members").reset_index()
    usage = pd.DataFrame([dict(Members=len(q), AverageWeeklyAttendance=q.AverageFrequency.mean(),
                              AverageFinalWindowUtilisationPct=q.Ratio.mean(),
                              AverageActivityVariety=q.NumberOfActivities.mean())])
    full = pd.DataFrame({"MonthNumber":range(1,13)})
    for datecol, mask, label in (("StartDate",pd.Series(True,index=q.index),"FirstRegistrations"),
                                 ("FinishDate",q.Dropout.eq(1),"RecordedExits")):
        counts=q.loc[mask,datecol].dt.month.value_counts()
        full[label] = full.MonthNumber.map(counts).fillna(0).astype(int)
    full.insert(1,"MonthName",full.MonthNumber.map(lambda n:calendar.month_name[n]))
    return {"ActivityByAge":pd.DataFrame(rows), "ActivityVarietyByAgeGender":heat,
            "Renewals":renew, "Usage":usage, "FullHistoryMonths":full}
