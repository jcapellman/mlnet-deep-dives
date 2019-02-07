using Microsoft.ML.Data;

namespace mljpegartifactdetector.Structures
{
    public class JpegArtifactorDetectorPrediction
    {
        [ColumnName("Score")]
        public float ContainsJpegArtifacts;
    }
}