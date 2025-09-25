从以下邮件内容中提取新闻的title、url、summary、date、sender），并以JSON列表的方式输出。

要求：
- 请先阅读理解内容，只提取新闻信息,如果没有可以为空
- 提取邮件发送者的email，作为sender 
- 不做翻译,保持原语言


请确保：
- sender: newslater发送者
- title: 新闻的标题
- url: 相关新闻的URL链接（如果有）
- summary: 新闻的摘要，简洁概括新闻内容
- date: 新闻的发布日期，格式为yyyy-MM-dd hh:mm。如果新闻没有明确的日期，使用邮件的发送时间。

请按以下格式输出JSON：
```json
[
  {
    "title": "新闻标题",
    "url": "相关新闻的URL链接",
    "summary": "新闻的简洁摘要",
    "date": "新闻发布日期，格式为yyyy-MM-dd hh:mm",
    "sender": "新闻推送者"
  }
]
```
生成后，请严格校验json格式

邮件文本如下：


