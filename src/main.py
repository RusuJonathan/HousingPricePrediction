from src.models.model import Model
from src.data.data_loader import load_yaml, load_data, train_path, test_path, model_config_path, target
import pandas as pd


df = load_data(train_path)
x_test = load_data(test_path)
config = load_yaml(model_config_path)

x_train = df.drop(columns=[target])
y_train = df[target]

model = Model(config=config, n_trials=300)
model.optimize_hyperparameters(x_train, y_train)

model.fit(x_train, y_train)

predictions = model.predict(x_test)

output = pd.DataFrame({"Id": x_test.Id,
                       "SalePrice": predictions})

output.to_csv("submission.csv", index=False)