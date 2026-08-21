# Voice2Latex
针对学生开发的语音助手。把老师上课的语音实时转成文字，最终得到两份产物：

- `raw.md` —— **原稿**，带时间戳，逐句照录，不做任何改写
- `notes.tex` —— **LaTeX 笔记**，分章节、公式已 LaTeX 化，可直接 xelatex 编译

中间产物 `notes/*.md` 是逐段清洗的结果（Markdown），`notes.md` 是它们的拼接版。

## 工作原理

```
麦克风 / 系统音频
   │
   ├─ 流式 Paraformer ──────────────→ 终端实时字幕（低延迟，允许有错，不入库）
   │
   └─ VAD 切句 → SenseVoice 重识别 ─→ raw.jsonl（append-only，原稿的权威来源）
                                        │
                          每约 2 分钟一批，云端 LLM 清洗
                                        ↓
                                   notes/NNNN.md（去口头语、纠术语、公式转 LaTeX）
                                        │
                    下课后云端 LLM 整合一次 ←── 课件 PDF/PPTX（可选）
                                        ↓              文字用于校正符号术语
                                    正文 LaTeX          图片用于插入
                                        ↓
                                 再一次调用写「本讲概要」
                                        ↓
                                    notes.tex
```

## 文件结构

```
Voice2Latex/
├── lecture/                    主包
│   ├── __main__.py             `python -m lecture` 的入口
│   ├── cli.py                  子命令都在这里（live / notes / serve / video …）
│   ├── web.py                  网页界面的 HTTP 接口（serve）
│   ├── static/index.html       界面本体，单文件、无外部依赖
│   ├── engine.py               采集引擎：音频 → 原稿 → 后台清洗，线程安全地暴露状态
│   ├── audio.py                音频输入：麦克风 / WASAPI loopback / 文件回放
│   ├── asr.py                  双通道 ASR：流式 Paraformer 出字幕，SenseVoice 定稿
│   ├── refine.py               逐段清洗：口语原稿 → 术语正确、公式 LaTeX 化的片段
│   ├── latex.py                课后整合：片段 → 完整 notes.tex（含概要与三道自动校验）
│   ├── slides.py               课件抽取：PDF / PPTX → 每页文字 + 图
│   ├── compile.py              notes.tex → notes.pdf（xelatex 跑两遍）
│   ├── llm.py                  OpenAI 兼容客户端，Ollama 与 DeepSeek 走同一套
│   ├── models.py               模型下载：镜像轮换 + 断点续传 + 进度回报
│   ├── store.py                session 目录读写，原稿 append-only
│   └── config.py               config.yaml / .env 的解析
├── scripts/fetch_models.py     模型下载的命令行外壳，逻辑在 lecture/models.py
├── glossaries/example.txt      术语表示例（自己的那份不提交，见 .gitignore）
├── config.yaml                 配置
├── requirements.txt
├── models/                     fetch 下载到这里，约 477MB · 不提交
└── sessions/                   每节课一个目录，见下 · 不提交
```

产出：

```
sessions/2026-08-20_1403_线性代数_第3讲/
├── meta.json         标题、创建时间、音频源；用了术语表、暂停过的话也记在这里
├── raw.jsonl         每句一行，append-only —— 原稿的权威来源
├── raw.md            人可读原稿，带时间戳
├── audio.wav         16k 单声道录音（只有实时录音才有，转写现成文件不重复存一份）
├── notes/0001.md     逐段清洗的结果，一批一个文件
├── notes.md          上面那些的拼接版，想快速看就看这个
├── notes.tex         课后整合的 LaTeX 笔记
├── notes.pdf         自动编译的产物
├── figures/          从课件抽出来的图（用了 --slides 才有）
└── state.json        已清洗到第几句，用于断点续跑
```

## 安装

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # mac / Linux
pip install -r requirements.txt
```

另外：转写已有录音需要 **ffmpeg** 在 PATH 里；出 PDF 需要 **xelatex**（MiKTeX / TeXLive /
TinyTeX 之一）

### 下载模型

```bash
python -m lecture fetch         # 镜像自动轮换 + 断点续传
python -m lecture models        # 确认就位
```

下载约 1.5GB，解压后默认删掉 fp32 权重只留 int8（`--keep-fp32` 可保留），落盘约 477MB。

### 云端整合

`.env.example` 复制成 `.env`，填写 DeepSeek 密钥：

```
DEEPSEEK_API_KEY=sk-xxxx
```

## 用法

```bash
# 网页界面（推荐）：点按钮开始/暂停，字幕显示在页面上
python -m lecture serve            # 浏览器打开 http://127.0.0.1:8730

# 上课前自检：录 10 秒，看音量够不够、能不能出字
python -m lecture check

# 查看电脑支持的音频设备
python -m lecture devices

# 线下课：录麦克风
python -m lecture live --title "线性代数 第3讲"

# 网课：录电脑播放声音
python -m lecture live --title "机器学习 第5讲" --source loopback

# 课上按键：p 暂停/继续（课间用），q 或 Ctrl-C 结束

# 结束后生成笔记；带上课件会用来校正符号术语，图也会插进笔记
python -m lecture notes [--slides 第5讲.pdf]

# 转写已有录音
python -m lecture transcribe 录音.m4a --title "高等数学"

# 课程视频/录播 → 笔记
python -m lecture video 第3讲.mp4 --title "计算机视觉 第3讲" --glossary example
```

产出都在 `sessions/2026-08-20_1403_线性代数_第3讲/` 下面。
`notes` 会自动把没清洗完的部分补上，所以课上没跑清洗也没关系。

### PDF

若使用网页界面， `notes` 和 `video` 跑完会**自动编译**，产出 `notes.pdf`。
中文要 `xelatex` + `ctexart`。

编译跑两遍（第一遍才生成 `.aux`/`.toc`，目录在第二遍才有内容），且**必须在 session
目录里进行**——笔记里的图是 `figures/xxx.png` 这种相对路径，在别处编会找不到图。

若使用 cli，需要**手动编译**：

```bash
cd sessions/xxx && xelatex notes.tex
python -m lecture.compile sessions/xxx/notes.tex   # 或者用这个
```

## 配置

`config.yaml`：

| 项 | 说明 |
|---|---|
| `asr.device` | 录音设备，留空 = 系统默认。**系统默认不总是能用**——若无法录制，用 `devices` 查看编号再填写 |
| `asr.min_silence_duration` | 静音多久算一句结束。老师语速慢、喜欢停顿则调大（0.5），语速快调小（0.25） |
| `asr.max_speech_duration` | 一句最长多久强制切断 |
| `asr.provider` | `cpu` / `cuda`。 CPU 已经足够  |
| `refine.batch_seconds` | 多久内容清洗一批。调大更连贯但延迟高，调小反之 |
| `llm.refine_backend` | `deepseek`（默认）或 `ollama`。改 ollama 时 `refine_model` 要填本地模型名，本地不可用会按 `refine_fallback_to_cloud` 退回云端 |
| `llm.compose_model` | 整合模型。`deepseek-chat` 是别名，实际路由到 `deepseek-v4-flash` |
| `llm.thinking` | DeepSeek 思考链，默认关。 |

录音文件位置可以用环境变量 `LECTURE_SESSIONS_DIR` 覆盖，优先于 `config.yaml` 里的
`sessions_dir`。一节 90 分钟的课是 173MB wav、一学期约 27GB，建议不用则及时删除。

## 网页界面

```bash
python -m lecture serve            # 浏览器打开 http://127.0.0.1:8730
```

填课程标题和术语表 → 点「开始上课」→ 课间点「暂停」→ 下课点「结束」，然后填课件路径点
「生成」。实时字幕显示在页面上。界面里的操作和命令行等价，`sessions/` 下的
产出格式也一样。

## 课程视频 → 笔记

录播、Zoom 录像、下载的课程回放，`video` 一条命令走完转写 + 清洗 + 整合；界面上是最后
那张「课程视频 → 笔记」卡片。

- **格式**：ffmpeg 可读（mp4 / mkv / flv / mov / m4a / mp3 …），视频流被 `-vn` 丢掉。
- **任务在服务端跑**，关掉浏览器、断网都不影响，回来刷新页面就能看到进度。
- **一次一个**。视频任务和课堂录制抢同一个 ASR 模型，所以两者互斥。
- 不下载在线视频。要处理 B 站/慕课的课，需要先手动下载。

## 自检

`python -m lecture check` 录 10 秒，报告音量（RMS / 峰值）并当场识别一遍。能出字就说明
整条链路是通的。

### 课间暂停

课上按 `p` 暂停，再按 `p` 继续。暂停期间**既不录音也不识别**——课间的闲聊不会被转写进
原稿，也不会送去清洗。

时间戳因此是「录音里的位置」而非墙上时钟：`raw.md` 里的 `[12:34]` 永远能直接定位到
`audio.wav` 的 12 分 34 秒。暂停位置记在 `meta.json` 的 `pauses` 里。终端不是交互式的
时候（输出重定向、后台运行）按键功能自动关闭，只能用 Ctrl-C 结束。

## 课程术语表

术语纠正在 LLM 层做（sherpa-onnx 的热词只支持 transducer 模型，本项目用的两个都用不上）。
**这个文件对效果的影响比换模型大**，也是唯一需要长期维护的东西：看到笔记里术语错了就补一条。
一门课一个文件放在 `glossaries/` 下，用 `--glossary 文件名` 选（不带 `.txt`）：

```bash
python -m lecture live --title "线性代数 第3讲" --glossary example
python -m lecture notes --slides 第3讲.pdf        # 术语表已记在 meta.json，不用再指定
```

课程名会写进 `meta.json`，所以课后 `refine` / `notes` 会自动用同一份；不指定 `--glossary`
时用仓库根目录的 `glossary.txt`。仓库里只有 `glossaries/example.txt` 一份示例讲格式：

```
错的=>对的      # 把左边的识别错误纠正为右边
术语            # 正确写法白名单，告诉模型这个词该怎么写
# 井号开头是注释
```

文科比理科更依赖术语表：理科术语被上下文强约束，模型能反推；人名、地名、书名推不出来。

## 带课件一起总结

`notes --slides 课件.pdf`（或 `.pptx`）会把课件当参考资料，做两件事：

**1. 校正转写错误。** 符号和术语最容易听错，而课件上写得清清楚楚——老师说的学习率 α
被听成 "L"，纯靠转写会整合成 `\text{lr}` 甚至错认成损失 $L$，给了课件就能写对成 $\alpha$。

**2. 把课件里的图插进笔记。** 图抽到 `sessions/<x>/figures/`，模型在「看这个图」「如图」
这类指代处插入 `figure` 环境，路径只能从清单里复制、生成后还会检查文件是否真实存在。
内嵌位图直接抽（过滤图标和分割线），矢量元素够多的整页渲染成 PNG，纯文字页不出图。

课件只用于校正和配图，prompt 里明确禁止把老师没讲的内容搬进笔记；和录音内容不匹配时不会
硬插图。课件**刻意不喂给逐段清洗阶段**，只在整合阶段生效——那一步用的是能力更强的云端模型。
文字超过 9000 字会截断（每页最多取 320 字）。
