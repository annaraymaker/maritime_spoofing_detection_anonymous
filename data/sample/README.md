# Sample Data Placeholder

This directory would contain sample AIS data for testing, but actual data
has been withheld for the following reasons:

1. **Data Licensing**: The AIS data used in this research was obtained under
   commercial license from Spire Global and cannot be redistributed.

2. **Privacy Considerations**: AIS data contains vessel identification and
   movement patterns that may be commercially sensitive.

3. **File Size**: The full dataset comprises billions of position reports
   across 367,000+ vessels, totaling several terabytes.

## Obtaining Test Data

For reviewers who wish to test the pipeline, we suggest:

1. **Public AIS Sources**: Limited AIS data is available from:
   - Marine Cadastre (US waters): https://marinecadastre.gov/ais/
   - Danish Maritime Authority: https://dma.dk/safety-at-sea/navigational-information/ais-data

2. **Synthetic Data**: Generate synthetic vessel tracks using maritime
   simulation tools that follow realistic movement patterns.

3. **Request to Authors**: Contact the authors through the conference review
   system to request a small anonymized sample dataset.

## Expected Data Format

See `schemas/DATA_FORMATS.md` for the expected input data structure.
The pipeline expects gzipped JSON files with the following naming convention:

```
all_ships_routes_month{N}_{partition}.json.gz
```

Where `N` is the month number and `partition` is a numeric identifier for
data sharding.
