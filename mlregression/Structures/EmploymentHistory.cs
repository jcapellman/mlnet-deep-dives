﻿using mldeepdivelib.Common;

namespace mlregression.Structures
{
    public class EmploymentHistory
    {
        public string PositionName { get; set; }

        [Label(0, 150)]
        public float DurationInMonths { get; set; }

        public float IsMarried { get; set; }

        public float BSDegree { get; set; }

        public float MSDegree { get; set; }

        public float YearsExperience { get; set; }

        public float AgeAtHire { get; set; }

        public float HasKids { get; set; }

        public float WithinMonthOfVesting { get; set; }

        public float DeskDecorations { get; set; }
    }
}