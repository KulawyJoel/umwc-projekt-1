# AML Lab 01 — Pierwszy eksperyment (CLI v2)

## Uruchomienie
az extension add -n ml -y
az configure --defaults group=rg-aml-50545 workspace=amlws-50545 location=italynorth
az ml compute create --name cpu-cluster-50545 --type amlcompute --size Standard_DS2_v2 --min-instances 0 --max-instances 1 --idle-time-before-scale-down 120
az ml job create --file job.yml --stream

## Wyniki
- Metryki: accuracy (Studio → Experiments → intro-sklearn)
- Artefakty: `outputs/model.joblib`, `outputs/metrics.json`
