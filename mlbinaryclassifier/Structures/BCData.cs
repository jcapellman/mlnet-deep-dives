using Microsoft.ML.Data;

namespace mlbinaryclassifier.Structures
{
    public class BCData
    {
        [Column(ordinal: "0", name: "Label")]
        public float StillThere { get; set; }

        [Column(ordinal: "1")]
        public float LongCommute { get; set; }
        
        [Column(ordinal: "2")]
        public float PromotionLimited { get; set; }
        
        [Column(ordinal: "3")]
        public float Overwhelmed { get; set; }
    }
}