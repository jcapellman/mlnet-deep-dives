﻿using System.IO;

using Microsoft.ML;

namespace mldeepdivelib.Helpers
{
    public static class Predictor
    {
        public static TK Predict<T, TK>(MLContext mlContext, string modelPath, T predictionData) where T : class where TK : class, new()
        {
            ITransformer trainedModel;

            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
            }

            var predFunction = mlContext.Model.CreatePredictionEngine<T, TK>(trainedModel);

            return predFunction.Predict(predictionData);
        }

        public static TK Predict<T, TK>(MLContext mlContext, string modelPath, string predictionFilePath) where T : class where TK : class, new()
        {
            var data = File.ReadAllText(predictionFilePath);

            var predictionData = Newtonsoft.Json.JsonConvert.DeserializeObject<T>(data);

            return Predict<T, TK>(mlContext, modelPath, predictionData);
        }
    }
}