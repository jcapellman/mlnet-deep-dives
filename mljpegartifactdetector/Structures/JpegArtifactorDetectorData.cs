using Microsoft.ML.Data;

namespace mljpegartifactdetector.Structures
{
    public class JpegArtifactorDetectorData
    {
        [LoadColumn(0)]
        public float ContainsJpegArtifacts { get; set; }

        [Column("Data")]
        public int[] Data { get; set; }

        public override string ToString() => $"{ContainsJpegArtifacts}, {string.Join("\t", Data)}";
    }
}