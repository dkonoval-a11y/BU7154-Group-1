-- Four-entity sports-centre data model.
PRAGMA foreign_keys = ON;
CREATE TABLE Member (
    Member_ID VARCHAR(36) PRIMARY KEY NOT NULL,
    Age INTEGER NOT NULL,
    Gender VARCHAR(30) NOT NULL
);
CREATE TABLE Membership (
    Member_ID VARCHAR(36) PRIMARY KEY NOT NULL REFERENCES Member(Member_ID),
    EnrollmentStart DATETIME NOT NULL,
    EnrollmentFinish DATETIME NOT NULL,
    EnrollmentDuration INTEGER NOT NULL,
    AverageFrequency DECIMAL NOT NULL,
    UseByTime BOOLEAN NOT NULL CHECK(UseByTime IN (0,1)),
    LifetimeValue DECIMAL NOT NULL
);
CREATE TABLE MembershipStatus (
    Member_ID VARCHAR(36) PRIMARY KEY NOT NULL REFERENCES Membership(Member_ID),
    Dropout BOOLEAN NOT NULL CHECK(Dropout IN (0,1)),
    NumberOfRenewals INTEGER NOT NULL,
    AllowedNumberOfVisitsBySLA DECIMAL NOT NULL,
    RealNumberOfVisits INTEGER NOT NULL,
    Ratio DECIMAL NOT NULL
);
CREATE TABLE Activity (
    Activity_Name VARCHAR(80) NOT NULL,
    Member_ID VARCHAR(36) NOT NULL REFERENCES Membership(Member_ID),
    PRIMARY KEY(Activity_Name, Member_ID)
);
CREATE VIEW MemberSummary AS
SELECT m.Member_ID, m.Age, m.Gender, s.EnrollmentStart, s.EnrollmentFinish,
       s.EnrollmentDuration, s.AverageFrequency, s.UseByTime, s.LifetimeValue,
       u.Dropout, u.NumberOfRenewals, u.AllowedNumberOfVisitsBySLA,
       u.RealNumberOfVisits, u.Ratio
FROM Member m
JOIN Membership s USING(Member_ID)
JOIN MembershipStatus u USING(Member_ID);
