# covid19-india

## Repo Structure
    The repository is structured as follows:
    ./
    |-- config    : Various config params to be entered by humans go here
    |-- data      : All data go in here, files are tracked by git-lfs
    |-- outputs   : All output data go in here, files are tracked by git-lfs
    |-- docs      : All documentation other than README goes here
    |-- Makefile  : Makefile to clean data, run tests etc
    |-- notebooks : All notebooks go here
    |-- setup.sh  : Setup script to setup path etc
    `-- src       : All production code, resusable code components go here
        |-- configs        : All configs params go here
        |-- data_fetchers  : These are utilities for fetching data etc
        |-- driver.py      : Sample Code
        |-- entities       : Important entities, these are provided for data verification
        |-- model_wrappers : wraps main models, currently seir.py wraps the seirsplus models
            |-- base.py                            : has Abstract Base Class for wrapping models
            |-- model_factory.py                   : has model factory to build models
            |-- seir.py                            : SEIR model wrapper, which wraps around seirsplus model
            |-- intervention_enabled_model_base.py : General Wrapper for allowing interventions
            `-- intervention_enabled_seir.py       : Wrapper for seir model, only just maps variables
        |-- modules        : 
            |-- forecasting_module.py              : Makes prediction for a given region (and region type) for start-end date.
            |-- model_evaluator.py                 : Various evaluation utils, for_region, for various loss funcs etc
            |-- training_module.py                 : Training module, this currently used hyperopt to optimize
            |-- data_fetcher_module.py             : Data fetcher module for fetching various data into requisite format
            `-- scenario_forecasting_module.py     : Scenario based forecasting module.
        `-- utils          : Various tools like loss_funcs, hyperopt wrapper etc


## Class Structure

## Work-Flow
1. We follow [git-flow](https://nvie.com/posts/a-successful-git-branching-model/) with staging in our parlance being develop in theirs.
2. Ensure that your code satisfies [pep8](https://www.python.org/dev/peps/pep-0008/)
3. Ensure that you have documentation as per [pep257](https://www.python.org/dev/peps/pep-0257/) 
4. TODO: Decide PEP257, PEP287, something else.

## Adding new models to Notebooks (notebooks)
1. Checkout a branch from staging called feature/nb_(branch_name).
2. Either write your own models in (labs) and import in your notebook. If this a pypi, then add to your requirements file. If you are modidying an external folder, then add the external model folder inside extern_libs.
3. Now you can run your algorithms, provided you can marshall our data into what your model requires.
4. TODO: Need to expand this and verify this.

## Adding new model wrapper to Production (src)
1. If you need new parameters, then update that in entities in forecast_variables. 
2. If you are adding new model
    a. Then add that in model_class.py
    b. Create your new_model.py in model_wrappers and update the model_factory
3. if you want this model to be intervention enabled, then write the wrapper that maps variables call it intervention_enabled_your_model.py and place in model_wrappers.
4. TODO: Add more here

#### Data References
Country | Data Source
--- | --- 
US | https://www.kaggle.com/sudalairajkumar/covid19-in-usa 
Italy | https://www.kaggle.com/sudalairajkumar/covid19-in-italy
Global Data | https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
Global recovered data | https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv
