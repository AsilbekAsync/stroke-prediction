import pandas as pd
from colorama import Fore, Style, init
from brain import predict_new_patient

init(autoreset=True)

df = pd.read_csv("data/input.csv")
results = []

ok = 0
total = 0

for idx, row in df.iterrows():
    total += 1
    data = row.drop("id").to_dict()
    result = predict_new_patient(data)
    prediction = result["prediction"]
    checked = data["stroke"]

    if prediction == checked:
        ok += 1
        print(f"{Fore.GREEN}[OK]{Style.RESET_ALL}  id={row['id']}: {prediction}")
    else:
        print(f"{Fore.RED}[BAD]{Style.RESET_ALL} id={row['id']}: {prediction}, expected {checked}")

    results.append({**data, **result})

pd.DataFrame(results).to_csv("prediction_results.csv", index=False)
print(f"\nAccuracy: {ok}/{total} = {ok / total:.2%}")
