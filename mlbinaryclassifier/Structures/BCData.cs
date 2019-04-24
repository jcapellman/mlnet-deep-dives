using Microsoft.ML.Data;

namespace mlbinaryclassifier.Structures
{
    public class BCData
    {
        [LoadColumn(0)]
        public float IsHappy { get; set; }

        [LoadColumn(1)]
        public string Content { get; set; }
    }
}