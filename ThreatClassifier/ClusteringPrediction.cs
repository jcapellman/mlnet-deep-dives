using System;
using System.IO;
using System.Linq;
using System.Text;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Common;
using mldeepdivelib.Helpers;

using Microsoft.ML;

using ThreatClassifier.Common;
using ThreatClassifier.Structures;

namespace ThreatClassifier
{
    public class ClusteringPrediction : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<ThreatInformation>();

            var textReader = MlContext.Data.CreateTextReader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var dataView = textReader.Read(args[1]);

            var pipeline = MlContext.Transforms
                .Concatenate(Constants.FEATURE_COLUMN_NAME, modelObject.ToColumnNames())
                .Append(MlContext.Clustering.Trainers.KMeans(
                    Constants.FEATURE_COLUMN_NAME,
                    clustersCount: Enum.GetNames(typeof(ThreatTypes)).Length));

            var trainedModel = pipeline.Fit(dataView);

            using (var fs = File.Create(args[2]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[2]}");
        }

        protected override void Predict(string[] args)
        {
            var extraction = FeatureExtractFile(args[2], true);

            if (extraction == null)
            {
                return;
            }

            Console.WriteLine($"Predicting on {args[2]}:");

            var prediction = Predictor.Predict<ThreatInformation, ThreatPredictor>(MlContext, args[1], extraction);

            PrettyPrintResult(prediction);
        }

        protected override void FeatureExtraction(string[] args)
        {
            var startDate = DateTime.Now;

            var files = Directory.GetFiles(args[1]);

            var sb = new StringBuilder();

            foreach (var filePath in files)
            {
                var extraction = FeatureExtractFile(filePath);

                if (extraction == null)
                {
                    continue;
                }

                sb.AppendLine(extraction.ToString());
            }

            File.WriteAllText(args[2], sb.ToString());

            Console.WriteLine($"Feature Extraction completed in {DateTime.Now.Subtract(startDate).TotalMinutes} minutes to {args[2]}");
        }

        private void PrettyPrintResult(ThreatPredictor prediction)
        {
            var threatType = (ThreatTypes)prediction.ThreatClusterId;

            Console.WriteLine($"Threat Type: {threatType}");

            Console.WriteLine("Category Breakdown:");

            for (var x = 0; x < prediction.Distances.Length; x++)
            {
                threatType = (ThreatTypes)x;

                Console.WriteLine($"{threatType} - {prediction.Distances[x]}%");
            }
        }

        private ThreatInformation FeatureExtractFile(string filePath, bool forPrediction = false)
        {
            var peFile = new PeNet.PeFile(filePath);

            var information = new ThreatInformation
            {
                NumberImports = peFile.ImageResourceDirectory.NumberOfIdEntries,
                DataSizeInBytes = peFile.ImageSectionHeaders.FirstOrDefault()?.SizeOfRawData ?? 0.0f
            };

            foreach (var name in Enum.GetNames(typeof(ThreatTypes)))
            {
                if (!filePath.Contains(name))
                {
                    continue;
                }

                information.Classification = name;
            }

            if (!string.IsNullOrEmpty(information.Classification) || forPrediction)
            {
                return information;
            }

            Console.WriteLine($"{filePath} was not named properly");

            return null;
        }
    }
}