using Microsoft.ML.Runtime.Api;

namespace mlregression.Structures
{
    public class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float DurationInMonths;
    }
}