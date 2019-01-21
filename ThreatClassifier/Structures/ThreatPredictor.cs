using Microsoft.ML.Data;

namespace ThreatClassifier.Structures
{
    public class ThreatPredictor
    {
        [ColumnName("PredictedLabel")]
        public uint ThreatClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}