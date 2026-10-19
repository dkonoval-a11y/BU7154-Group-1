"""Tests for dates, age groups and cohort weights."""
import unittest
import numpy as np
import pandas as pd
from src.analysis import (ACTIVITIES, MASTER_COLUMNS, GROUP_A, monthly, prepare,
                          retention_cohorts, retention_summary)


def record(member, start, finish="2019-10-31", dropout=0, age=20):
    r={c:0 for c in MASTER_COLUMNS if c not in ("FirstAnniversaryDate","ObservedTenureDays")}
    r.update(Id=member,Age=age,Gender="Female",EnrollmentStart=pd.Timestamp(start),
             EnrollmentFinish=pd.Timestamp(finish),Dropout=dropout,
             LastPeriodStart=pd.Timestamp(start),LastPeriodFinish=pd.Timestamp(finish),
             FitnessActivities=1,NumberOfActivities=1,AllowedNumberOfVisitsBySLA=10,
             EnrollmentDurationClass1=1,EnrollmentDurationClass2="[00,01]",
             AverageFrequencyClass1=1,AverageFrequencyClass2="[0.0,0.1]",RatioClass2="[0] ")
    return r


class MeaningTests(unittest.TestCase):
    def test_same_month_exit_is_not_in_opening_risk_numerator(self):
        raw=pd.DataFrame([record("opening","2014-12-01","2015-01-01",1),
                          record("same_month","2015-01-15","2015-01-20",1),
                          record("continuing","2015-01-31")])
        _,d=prepare(raw);m=monthly(d).iloc[0]
        self.assertEqual((m.OpeningMemberships,m.NewEnrolments,m.Dropouts,m.ActiveMemberships),(1,2,2,1))
        self.assertEqual(m.DropoutsFromOpening,1)
        self.assertEqual(m.SameMonthEnrolmentDropouts,1)
        self.assertEqual(m.DropoutRate,1.0)

    def test_cutoff_is_not_a_dropout(self):
        _,d=prepare(pd.DataFrame([record("still_open","2014-07-01")]))
        m=monthly(d)
        self.assertEqual(m.Dropouts.sum(),0)
        self.assertTrue(m.ActiveMemberships.eq(1).all())

    def test_exit_on_day_90_and_inclusive_sensitivity(self):
        _,d=prepare(pd.DataFrame([record("boundary","2018-01-01","2018-04-01",1)]))
        c,_=retention_cohorts(d)
        ci,_=retention_cohorts(d,inclusive_final_day=True)
        select=lambda x:x[(x.CohortMonth==pd.Timestamp("2018-01-01"))&(x.AgeGroup==GROUP_A)&(x.HorizonDays==90)].iloc[0]
        self.assertEqual(select(c).ExitsByHorizon,1)
        self.assertEqual(select(c).RetainedN,0)
        self.assertEqual(select(ci).RetainedN,1)

    def test_early_failure_stays_in_denominator_but_early_censoring_does_not(self):
        _,d=prepare(pd.DataFrame([record("failure","2018-01-01","2018-01-10",1),
                                 record("censored","2018-01-01","2018-02-01",0)]))
        c,_=retention_cohorts(d)
        r=c[(c.CohortMonth==pd.Timestamp("2018-01-01"))&(c.AgeGroup==GROUP_A)&(c.HorizonDays==90)].iloc[0]
        self.assertEqual((r.CohortSize,r.EligibleN,r.ExitsByHorizon,r.CensoredBeforeHorizon),(2,1,1,1))

    def test_midpoint_age_and_unknown_age_scope(self):
        from src.analysis import age_masks
        _,d=prepare(pd.DataFrame([record("unknown","2014-07-01",age=0),record("older","2014-07-01",age=40)]))
        self.assertEqual(len(d),2)
        self.assertFalse(d.ValidAge.iloc[0])
        expected=40.5-(pd.Timestamp("2019-10-31")-pd.Timestamp("2014-07-01")).days/365.2425
        self.assertAlmostEqual(d.EstimatedEntryAge.iloc[1],expected)
        self.assertEqual(sum(int(x.sum()) for x in age_masks(d).values()),1)

    def test_same_weights_for_both_groups(self):
        p=pd.DataFrame(dict(HorizonDays=[90,90],EligibleNA=[100,10],EligibleNB=[10,100],
                            RetainedNA=[90,1],RetainedNB=[10,40],RetentionRateA=[.9,.1],
                            RetentionRateB=[1.,.4],GapPP_AminusB=[-10.,-30.],
                            PairedSupport=[True,True],SmallEitherGroup=[True,True],
                            PooledCohortWeightN=[110,110]))
        r=retention_summary(p,"example")
        self.assertAlmostEqual(r["MatchedRateA"],.5)
        self.assertAlmostEqual(r["MatchedRateB"],.7)
        self.assertAlmostEqual(r["RawRateA"],91/110)
        self.assertAlmostEqual(r["RawRateB"],50/110)
        self.assertAlmostEqual(r["GapPP_AminusB"],-20)

    def test_excel_calendar_helpers_keep_dates_without_time(self):
        master,_=prepare(pd.DataFrame([record("leap","2016-02-29 23:15","2017-03-01 01:00",1)]))
        self.assertEqual(master.FirstAnniversaryDate.iloc[0],pd.Timestamp("2017-02-28"))
        self.assertEqual(master.ObservedTenureDays.iloc[0],366)


if __name__=="__main__":unittest.main()
