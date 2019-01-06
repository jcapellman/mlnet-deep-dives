using System;

namespace mldeepdivelib.Common
{
    [AttributeUsage(AttributeTargets.Property)]
    public class LabelAttribute : Attribute
    {
        public int Min { get; set; }

        public int Max { get; set; }

        public LabelAttribute(int min, int max)
        {
            Min = min;
            Max = max;
        }
    }
}