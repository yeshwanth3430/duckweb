from db_operations import delete_rows_by_date

# Specify the date to delete
DATE_TO_DELETE = '2024-01-05 '

# Delete from NIFTY 1min
result_nifty = delete_rows_by_date('nifty_1min', DATE_TO_DELETE)
print(f'NIFTY 1min deletion success: {result_nifty}')

# Delete from BANKNIFTY 1min
result_banknifty = delete_rows_by_date('banknifty_1min', DATE_TO_DELETE)
print(f'BANKNIFTY 1min deletion success: {result_banknifty}') 