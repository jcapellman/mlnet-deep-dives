using Microsoft.ML.Data;

namespace mlregression.Structures
{
    public class EmploymentHistoryPrediction
    {
        [ColumnName("Score")]
        public float DurationInMonths;
    }
}