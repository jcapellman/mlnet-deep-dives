using Microsoft.ML.Data;

namespace mlbinaryclassifier.Structures
{
    public class BCData
    {
        [Column(ordinal: "0", name: "Label")]
        public float IsHappy { get; set; }

        [Column(ordinal: "1")]
        public string Content { get; set; }
    }
}