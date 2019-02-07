using mldeepdivelib.Common;

namespace mljpegartifactdetector.Structures
{
    public class JpegArtifactorDetectorData
    {
        [Label(int.MinValue, int.MaxValue)]
        public float ContainsJpegArtifacts { get; set; }

        public byte[] Data { get; set; }

        public override string ToString() => $"0.0f, {System.Text.Encoding.UTF8.GetString(Data)}";
    }
}