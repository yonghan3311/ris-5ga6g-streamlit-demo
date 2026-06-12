# RIS 5G-A/6G Streamlit Demo

这是“面向 5G-A/6G 的 RIS 智能反射面辅助毫米波蜂窝覆盖增强与干扰优化仿真”的 Streamlit Community Cloud 部署版本。

## 文件结构

- `app.py`：Streamlit 网页入口文件。
- `src/ris_simulator.py`：RIS 毫米波覆盖、SINR、吞吐率等仿真核心。
- `requirements.txt`：云端自动安装的 Python 依赖。
- `.streamlit/config.toml`：网页主题配置。

## 本地预览

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

部署时将本目录内容作为 GitHub 仓库根目录，Main file path 填写：

```text
app.py
```
