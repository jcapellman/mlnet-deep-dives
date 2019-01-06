using System.Linq;
using System.Reflection;

using Microsoft.ML.Runtime.Data;

namespace mldeepdivelib.Common
{
    public static class ExtensionMethods
    {
        public static TextLoader.Column[] ToColumns<T>(this T model)
        {
            var fields = model.GetType().GetProperties(BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.Instance);

            return fields.Select((t, x) => new TextLoader.Column(t.Name, t.PropertyType == typeof(string) ? DataKind.Text : DataKind.R4, x)).ToArray();
        }
    }
}