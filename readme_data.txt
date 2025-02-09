Source:
https://www.kaggle.com/datasets/kornilovag94/bank-credit-default-loan-default/
Files:
train_data_0.pq, train_data_1.pq - input variables + key (parquet format)
target.csv - output variable + key (CSV, comma sep)

Data description (input variables):
id - identifier of the application (key for joining tables)
rn - sequence number of the credit product in the credit history
pre_since_opened - days from credit opening date to data collection date
pre_since_confirmed - days from credit information confirmation date till data collection date
pre_pterm - planned number of days from credit opening date to closing date
pre_fterm - actual number of days from credit opening date to closing date
pre_till_pclose - planned number of days from data collection date until loan closing date
pre_till_fclose - actual number of days from data collection date until loan closing date
pre_loans_credit_limit - credit limit
pre_loans_next_pay_summ - amount of the next loan payment
pre_loans_outstanding - outstanding loan amount
pre_loans_total_overdue - current overdue amount
pre_loans_max_overdue_sum - maximum overdue amount
pre_loans_credit_cost_rate - total cost of credit
pre_loans5 - number of delinquencies of up to 5 days
pre_loans530 - number of delinquencies from 5 to 30 days
pre_loans3060 - number of delinquencies from 30 to 60 days
pre_loans6090 - number of delinquencies from 60 to 90 days
pre_loans90 - number of delinquencies of more than 90 days
is_zero_loans_5 - flag: no delinquencies of up to 5 days
is_zero_loans_530 - flag: no delinquencies of 5 to 30 days
is_zero_loans_3060 - flag: no delinquencies of 30 to 60 days
is_zero_loans_6090 - flag: no delinquencies of 60 to 90 days
is_zero_loans90 - flag: no delinquencies of more than 90 days
pre_util - ratio of outstanding loan amount to credit limit
pre_over2limit - ratio of currently overdue debt to credit limit
pre_maxover2limit - ratio of maximum overdue debt to credit limit
is_zero_util - flag: ratio of outstanding loan amount to credit limit equals 0
is_zero_over2limit - flag: ratio of current overdue debt to credit limit equals 0
is_zero_maxover2limit - flag: ratio of maximum overdue debt to credit limit equals 0
enc_paym_{0…n} - monthly payment statuses of the last n months
enc_loans_account_holder_type - type of relation to the loan
enc_loans_credit_status - credit status
enc_loans_account_cur - currency of the loan
enc_loans_credit_type - credit type
pclose_flag - flag: planned number of days from opening date to closing date of the loan
fclose_flag - flag: actual number of days from credit opening date to closing date undefined Usability

