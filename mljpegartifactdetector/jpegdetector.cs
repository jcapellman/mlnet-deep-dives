using System;
using System.IO;
using System.Text;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Common;
using mldeepdivelib.Helpers;

using mljpegartifactdetector.Structures;

using Microsoft.ML;
using Microsoft.ML.Transforms.Normalizers;

namespace mljpegartifactdetector
{
    public class jpegdetector : BaseMLPrediction
    {
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
        
        protected override void Train(string[] args)
        {
            var modelObject = Activator.CreateInstance<JpegArtifactorDetectorData>();

            var textReader = MlContext.Data.CreateTextLoader(columns: modelObject.ToColumns(), hasHeader: false, separatorChar: ',');

            var baseTrainingDataView = textReader.Read(args[1]);

            var label = modelObject.GetLabelAttributes();

            var trainingDataView = MlContext.Data.FilterByColumn(baseTrainingDataView, label.Name, label.Min, label.Max);

            var dataProcessPipeline = MlContext.Transforms.CopyColumns(label.Name, "Label")
                .Append(MlContext.Transforms.Normalize("Data", mode: NormalizingEstimator.NormalizerMode.MeanVariance))
                .Append(MlContext.Transforms.Concatenate("Features", "Data"));

            var trainer = MlContext.Regression.Trainers.StochasticDualCoordinateAscent();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            var trainedModel = trainingPipeline.Fit(trainingDataView);

            using (var fs = File.Create(args[2]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[2]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<JpegArtifactorDetectorData, JpegArtifactorDetectorPrediction>(MlContext, args[1], args[2]);

            Console.WriteLine($"Has Jpeg Artifacts: {prediction.ContainsJpegArtifacts:0.#}");
        }

        private JpegArtifactorDetectorData FeatureExtractFile(string filePath, bool forPrediction = false) =>
            new JpegArtifactorDetectorData
            {
                Data = File.ReadAllBytes(filePath)
            };
    }
}