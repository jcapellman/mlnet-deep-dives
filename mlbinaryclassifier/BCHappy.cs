using System;
using System.IO;

using mlbinaryclassifier.Structures;

using mldeepdivelib.Abstractions;
using mldeepdivelib.Helpers;

using Microsoft.ML;
using Microsoft.ML.Data;

namespace mlbinaryclassifier
{
    public class BCAttrition : BaseMLPrediction
    {
        protected override void Train(string[] args)
        {
            var textReader = MlContext.Data.CreateTextLoader(new TextLoader.Arguments
            {
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("Label", DataKind.Bool, 0),
                    new TextLoader.Column("Content", DataKind.Text, 1)
                }
            });

            var baseTrainingDataView = textReader.Read(args[1]);

            var pipeline = MlContext.Transforms.Text.FeaturizeText("Content", "Features")
                .Append(MlContext.BinaryClassification.Trainers.FastTree(numLeaves: 2, numTrees: 10, minDatapointsInLeaves: 1));

            var trainedModel = pipeline.Fit(baseTrainingDataView);

            using (var fs = File.Create(args[2]))
            {
                trainedModel.SaveTo(MlContext, fs);
            }

            Console.WriteLine($"Saved model to {args[2]}");
        }

        protected override void Predict(string[] args)
        {
            var prediction = Predictor.Predict<BCData, BCPrediction>(MlContext, args[1], args[2]);

            Console.WriteLine($"Likely not happy: {prediction.Probability * 100:##.#}%");
        }

        protected override void FeatureExtraction(string[] args)
        {
            throw new NotImplementedException();
        }
    }
}