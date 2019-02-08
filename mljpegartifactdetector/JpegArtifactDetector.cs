using System;
using System.IO;
using System.Text;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Helpers;

using mljpegartifactdetector.Structures;

using Microsoft.ML;

namespace mljpegartifactdetector
{
    public class JpegArtifactDetector : BaseMLPrediction
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
            var trainingDataView = MlContext.Data.ReadFromTextFile<JpegArtifactorDetectorData>(args[1], hasHeader: true);
            
            var dataProcessPipeline = MlContext.Transforms.Concatenate("Features", "Data").AppendCacheCheckpoint(MlContext);

            var trainer = MlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent(labelColumn: "Label", featureColumn: "Features");
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
                Data = System.Text.Encoding.UTF8.GetString(File.ReadAllBytes(filePath))
            };
    }
}