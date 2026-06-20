#EXPLANATION

#Question 1 ) What percentage of customers in your dataset have y = yes? What does this imbalance mean for how you'd evaluate a model?
in the given dataset i found a very huge imbalance between the people who subscribed and those who did not subscribed.
exact figures were as following :-
  yes : 5,289        -> 11.70%
  no  : 39,922       -> 88.30%
  total entries in raw data : 45,211
this class bias was huge and to handle it required alot of brainstorming at first but eventually figured out a way to make the things work . First i chose the baseline model Logistic Regression which was
compulsary to be done. the results were a little misleading as the accuracy = 89.86, precision = 63.53, recall = 31.29 and f1 = 41.93. Accuracy and precision were really good but they were misleading because the True positives were only 331.
in short that meant model was good at predicting if we submit it as a project but for industry level use case it was waste because it meant it could not identify enough customers who will subscribe and for those whom are found, we have to waste alot of calls.
this thing happened because of class bias.
to solve this issue i rather went out of the project description and trained 3 models (Random Forest, XG Boost, CatBoost) along with Logistic regression to find and experiment on how to make it a reall industry level tool.



# QUESTION 2 ) Which job category had the highest subscription rate? Does this make sense to you intuitively?
From the analysis of the RAW data i found that the student category had the highest subscription rate of 28.68%, followed by retired customers of 22.79%.
The results were quite interesting because students are not usually considered the wealthiest customer segment and the least likely to be engaged in things such as loans and subscriptions.
However, when i further trained moels and found the results, i noticed the pattern that the set of categories were making sense somehow because they can't spend much because they are not earning. in order to have more finds to invest makes sense.
and when things were off for me thats when i realized why we are working on this problem.
The second-highest category, retired customers, made intuitively much more sense to me. Retired individuals generally prefer loweer financial risks and are often more focused on preserving their savings rather than taking aggressive investment risks.
Since term deposits provide stability and predictable returns, they naturally appeal to this customer group.
On the other hand, categories such as blue-collar workers and entrepreneurs showed much lower subscription rates. One possible reason is that these groups may either have tighter financial constraints or prefer keeping their money available for day-to-day business and personal expenses rather than locking it into a term deposit.
Overall, I believe the results are reasonable and align with the financial behavior typically associated with these customer groups.
i have some other thoughts also at some places but again then only we realize why this model is useful and why its worth to work on.



# QUESTION 3 ) Which feature had the highest importance in your tree-based model? Why do you think that is?
To deal with the class imbalance present in the dataset, I experimented with multiple tree-based models including Random Forest, XGBoost and CatBoost. The goal was not simply to maximize accuracy, but to find a model that could identify actual subscribers while keeping unnecessary calls under control.
Among all the models, CatBoost was selected as the final model. According to the feature importance analysis, the most important feature was duration, with an importance score of approximately 32.3%.
At first, this result felt a little strange to me because duration is simply the length of the last call between the customer and the bank representative. However, after looking deeper into the data and comparing multiple models, it started making a lot more sense. Customers who spend more time
talking are usually more engaged in the conversation, ask more questions, and show greater interest in the offered product. Naturally, such customers are more likely to subscribe.
What I found interesting was that duration remained one of the most important features across Random Forest, XGBoost and CatBoost. Seeing the same pattern repeatedly increased my confidence that this was not just a model-specific result but a genuine business signal.
There were moments during the project when some results initially seemed counterintuitive. However, that is exactly why problems like this are worth working on. Human intuition can identify patterns, but data often reveals relationships that are not immediately obvious. The role of the model is
to capture those patterns consistently and use them to support better business decisions.



# Question  4 ) Why is F1 a better metric than accuracy for this particular dataset?
For this dataset, I believe F1-score is a much better metric than accuracy because of the heavy class imbalance present in the data. Only about 11.7% of customers actually subscribed to the term deposit, while the remaining 88.3% did not.
Because of this imbalance, a model can achieve a very high accuracy simply by predicting most customers as "No". In fact, if a model predicts every customer as "No", it would still achieve around 88% accuracy, but it would completely fail at the actual business objective of identifying potential subscribers.
During my experimentation with Logistic Regression, Random Forest, XGBoost, and CatBoost, I noticed that accuracy often remained high even when the model was missing a large number of actual subscribers. This made me realize that accuracy alone was not telling the complete story.
F1-score was more useful because it combines both precision and recall into a single metric. Precision helps measure how many predicted subscribers were actually subscribers, while recall measures how many real subscribers the model was able to identify. Since the bank wants to identify potential customers without making too many unnecessary calls,
both of these factors are important. F1-score helped me evaluate that balance much more effectively than accuracy alone.



# Question 5 ) Pick one of your 5 sample predictions. Do you actually agree with the model's call, given that customer's features? Walk through your thinking.
For this question, I chose the borderline customer from my sample predictions. The customer was 77 years old, retired, had a positive account balance, and had previous interactions with the bank. The model assigned a probability very close to my final decision threshold of 0.7 and ultimately predicted that the customer would not subscribe.
At first, I was not completely convinced by the prediction because some of the customer's characteristics appeared favorable. During my EDA, retired customers had one of the highest subscription rates, and customers with higher balances generally showed a greater tendency to subscribe. Based on those observations alone, I would have expected the customer to be a reasonable candidate.
However, after looking at the probability score and the final prediction, I understood why the model was uncertain. The customer's profile likely contained a mixture of positive and negative indicators, causing the prediction to fall very close to the decision boundary. Rather than making an extremely confident prediction, the model effectively treated this customer as a borderline case.
Overall, I agree with the model's decision. Even though some features suggested a possibility of subscription, the prediction was not made with overwhelming confidence. This is exactly the type of situation where a machine learning model can be useful, as it combines information from multiple features simultaneously instead of relying on a few intuitive observations.
