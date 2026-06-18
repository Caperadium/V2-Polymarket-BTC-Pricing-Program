AttributeError: 'list' object has no attribute 'empty'
Traceback:

File "C:\Users\Kieran Trythall\AppData\Local\Programs\Python\Python312\Lib\site-packages\streamlit\runtime\scriptrunner\script_runner.py", line 600, in _run_script
    exec(code, module.__dict__)
File "C:\Users\Kieran Trythall\Documents\Trading\Prediction Market Contract Pricing\V2 BTC Contract Pricing\app\pages\backtesting.py", line 195, in <module>
    n_new = orch.fetch_historical_prices()
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Kieran Trythall\Documents\Trading\Prediction Market Contract Pricing\V2 BTC Contract Pricing\core\backtesting\orchestrator.py", line 77, in fetch_historical_prices
    added = fetch_incremental_prices(
            ^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Kieran Trythall\Documents\Trading\Prediction Market Contract Pricing\V2 BTC Contract Pricing\core\backtesting\polymarket_fetcher.py", line 518, in fetch_incremental_prices
    n_added = store.append_incremental(all_new_records)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\Kieran Trythall\Documents\Trading\Prediction Market Contract Pricing\V2 BTC Contract Pricing\core\backtesting\contract_store.py", line 154, in append_incremental
    if new_records.empty:
       ^^^^^^^^^^^^^^^^^