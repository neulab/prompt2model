# Tips and Examples to Write a Good Prompt

## How to Write a Good Prompt

A good prompt can make the generated dataset
follow exactly the format of demonstrations.
It contains the instruction and few-shot examples.

The instruction should contain the following:

1. The exact format description for the input
and output, i.e., a string, a dictionary, or whatever.
2. The exact contents of each part of the
input and their relationship as possible as you can.
3. The range of possible input. For example,
"And the question can range from Math, Cultural,
Social, Geometry, Biology, History, Sports, Technology,
Science, and so on."

The few-shot examples should contain the following:

1. Use `=` rather than other ambiguous symbols like `:`.
2. Avoid unnecessary line breaks at the beginning.
For example, `input=""` is better than breaking
the line after `=`.
3. Use `input` rather than `Input`, `output` is
preferable likewise.
4. Wrap the `input` and `output` into a string with `“”`.

Though the examples are optional, we strongly
suggest including them to guide the format and
content for the generator.

Also, we recommend providing several precise examples
in the specified format and inquiring with ChatGPT
about the format and scope of your examples.

## Examples of Good Prompts

Here are some examples of good prompts:

### Question Answering

```text
"""Your task is to generate an answer to a natural question. In this task, the input is a string that consists of both a question and a context passage. The context is a descriptive passage related to the question and contains the answer. And the question can range from Math, Cultural, Social, Geometry, Biology, History, Sports, Technology, Science, and so on.

Here are examples with input questions and context passages, along with their expected outputs:

input="Question: What city did Super Bowl 50 take place in? Context: Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50."
output="Santa Clara"

input="Question: What river runs through Warsaw? Context: Warsaw (Polish: Warszawa [varˈʂava] ( listen); see also other names) is the capital and largest city of Poland. It stands on the Vistula River in east-central Poland, roughly 260 kilometres (160 mi) from the Baltic Sea and 300 kilometres (190 mi) from the Carpathian Mountains. Its population is estimated at 1.740 million residents within a greater metropolitan area of 2.666 million residents, which makes Warsaw the 9th most-populous capital city in the European Union. The city limits cover 516.9 square kilometres (199.6 sq mi), while the metropolitan area covers 6,100.43 square kilometres (2,355.39 sq mi)."
output="Vistula River"

input="Question: The Ottoman empire controlled territory on three continents, Africa, Asia and which other? Context: The Ottoman Empire was an imperial state that lasted from 1299 to 1923. During the 16th and 17th centuries, in particular at the height of its power under the reign of Suleiman the Magnificent, the Ottoman Empire was a powerful multinational, multilingual empire controlling much of Southeast Europe, Western Asia, the Caucasus, North Africa, and the Horn of Africa. At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the empire, while others were granted various types of autonomy during the course of centuries."
output="Europe"
"""
```

### Temporal Expression Normalization

```text
"""Temporal date expressions are commonly used to refer to specific time periods. Your task is to identify these temporal date expressions and provide the exact dates they refer to.

For this task, the input is a string containing two specific elements: a posted date in the format "[Posted: YYYY-MM-DD]" and a sentence or statement that contains various temporal date references (e.g., early December, the end of the year, today, August, last Christmas, next Month, etc).

Your program should output a string that maps the time period references mentioned in the input to their corresponding dates, following these strict rules:

1. If temporal date references are found, the output should use either "YYYY-MM-DD", "YYYY-MM", or "YYYY" to represent the exact date.
- If multiple time period references are found, separate them using '|'.
2. If no temporal date reference is found or the referred date is ambiguous, the output should just be 'N/A', i.e., output="N/A".

Here are some examples:

input="[Posted: 1998-09-07] Tourism industry revenues reportedly dropped to $300 million last year, down from $450 million the year before."
output="last year == 1997"

input="[Posted: 2013-09-27] Eat! @mepangilinan"
output="N/A"

input="[Posted: 1989-10-30] Rated single-B-1 by Moody's Investors Service Inc. and single-B-plus by Standard amp Poor's Corp., the issue will be sold through underwriters led by Goldman, Sachs amp Co. Hertz Corp. -- $100 million of senior notes due Nov. 1, 2009, priced at par to yield 9%."
output="Nov. 1, 2009 == 2009-11-01"

input="[Posted: 2014-07-11] So out of place with this early transfer business."
output="N/A"

input="[Posted: 2013-10-25] Quote of the Day: '#Yoga is what you learn on your way down!"
output="the Day == 2013-10-25"

input="[Posted: 2021-06-15] Google plans to develop PALM 2 model in the first quarter of next year."
output="N/A"

input="[Posted: 2013-03-22] We will release a new github repository in the next three months."
output="the next three month == 2013-04"

input="[Posted: 2022-05-17] The company's fiscal year starts on July 1st and ends on June 30th."
output="July 1st == 2022-07-01 | June 30th == 2022-06-30"

input="[Posted: 2013-03-22] This flu season started in early December, a month earlier than usual, and peaked by the end of year."
output="N/A"

input="[Posted: 1989-10-30] The issue, which is puttable back to the company in 1999, was priced at a spread of 110 basis points above the Treasury's 10-year note."
output="1999 == 1999"

input="[Posted: 2022-04-15] The company announced that they will release their new product at the end of next month."
output="the end of next month == 2022-05-31"

input="[Posted: 2022-03-15] The teacher is going to release a new assignment in a few days."
output="N/A"
"""
```

### Japanese-to-Python Generation

```text
"""Pythonで1行のコードを生成し、StackOverflowの日本語の質問を解決してください。コメントや式は含めないでください。インポート文も不要です。

このタスクでは、入力は日本語のテキストで、変数名や操作が記述されています。出力は、そのタスクを達成するためのPythonの1行のコードです。コメントや式は含めないでください。インポート文も不要です。

input="スペースで区切られた入力`stdin`を変数に格納して表示する"
output="for line in stdin: a = line.rstrip().split(' ') print(a)"

input="リスト`word_list'内に出現する単語を数える"
output="Counter(word_list)"

input="tweepyインスタンス`api`を使い、文字列`word`を含んだツイートを検索し、結果をリストとして得る"
output="search = api.search(q=word)"

input="データベースの設定を表示する"
output="print(settings.DATABASES)"

input="ネストされているリスト`li`を見やすく表示する"
output="pprint.pprint(li)"

input="HTMLファイル'test.html'を開き、テキストオブジェクト'text'をutf-8で保存する"
output="f = open('test.html', 'w') f.write(text.encode('utf-8'))"
"""
```
