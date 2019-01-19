using Microsoft.ML.Data;

using mldeepdivelib.Common;

namespace ThreatClassifier.Structures
{
    public class ThreatPredictor
    {
        [Label(0, 150)]
        public uint ThreatClusterId { get; set; }

        [ColumnName("Score")]
        public float[] Distances { get; set; }
    }
}