using Microsoft.ML.Data;

namespace ml_random_file_classification.Structures
{
    public class FilePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}