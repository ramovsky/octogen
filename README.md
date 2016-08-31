# octogen

1. I've matched dialogs agains TF-IFD index and LDA topic index. It works not well, cuz replics are to short. Also tried matching NER -- matching performance improved vastly, but calculations is more then 2hours.

2. Tested how it works on train dialogs. Current approach is disaster and I don't have much time to improve it.

3. I comme up with idea to detect questions "Does, Are...?" and match them with short answers eg. "Yes", "NO", "Of course"
