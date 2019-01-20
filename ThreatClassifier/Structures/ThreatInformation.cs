namespace ThreatClassifier.Structures
{
    public class ThreatInformation
    {
        public float NumberImports { get; set; }

        public float DataSizeInBytes { get; set; }

        public string Classification { get; set; }

        public override string ToString() => $"{Classification},{NumberImports},{DataSizeInBytes}";
    }
}