using Microsoft.ML.Data;

namespace mljpegartifactdetector.Structures
{
    public class JpegArtifactorDetectorData
    {
        [LoadColumn(0)]
        public float ContainsJpegArtifacts { get; set; }

        [LoadColumn(1)]
        public string FilePath { get; set; }

        [LoadColumn(2)]
        public int[] Data { get; set; }
    }
}