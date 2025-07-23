# Nexora - Large Language Models in the Shell

## Overview

Nexora is a command line utility that integrates Large Language Models (LLMs) into your local shell environment. It provides system access through a variety of tools to an LLM of choice, allowing you to perform various tasks with natural language. 

This program provides the LLM with tools enabling full system access, network access, and shell access. Any potentially dangerous tools, such as writing files or shell access, is behind a permission prompt for the user.

## Features

- **Natural Language Interface**: Perform various tasks using natural language commands
- **LLM Integration**: Compatible with multiple Large Language Models
- **System Access**: Provides LLMs tools for:
  - Full filesystem access
  - Network access
  - Shell access
- **Safety Measures**:
  - Permission prompts for potentially dangerous actions
  - Detailed previews of actions before execution
- **Versatile Tool Integration**: Enhances LLM capabilities with various system tools

## Example Uses

- **File Management**: "Find all PDF files created last month and move them to a new folder."
- **Data Analysis**: "Analyze the CSV file in my Downloads folder and generate a summary report."
- **System Maintenance**: "Check for and install system updates, then reboot if necessary."
- **Code Assistance**: "Find all TODO comments in my Python project and create a task list."
- **Network Diagnostics**: "Run a network speed test and log the results."
- **Text Processing**: "Extract all email addresses from the given text file and save them to a new file."
- **Automation**: "Create a bash script that backs up my Documents folder daily."

## Installation
To install Nexora from the Git repository, use the following commands:

```
pip3 install wheel
pip3 install git+https://github.com/bandirevanth/nexora
```

## Getting Started

Once installed, run `nexora` once to create your configuration in `~/.config/nexora`. Edit the configuration to specify your default LLM and, if applicable, API keys. Run `nexora` again to try it out.

Nexora currently supports OpenAI and Anthropic, with support for more providers (including Ollama) coming soon. 

For working within a specific directory, provide the path like so:

```
nexora path_to_dir
```

Upon starting Nexora, an interactive session is initiated, allowing the LLM to interact with the directory specified. Utilize the LLM's capabilities to manage tasks, ask questions, or perform actions within the shell environment.

## Examples

Explore a few examples of how Nexora can enhance your daily tasks:

- **File Management:** Instruct the LLM to rename, move, or modify files based on specific criteria.
- **Software Development:** Start Nexora within a project's directory, and ask it to make code revisions. Most of Nexora itself was written this way.
- **Data Analysis:** Leverage the LLM for data analysis, sorting, and summarization tasks.
- **Terminal Operations:** Get assistance and guidance from the LLM for executing commands and complex operations in the shell.

## Project Background

Nexora is an AI-driven project that showcases the capabilities of AI-generated content. The tool itself was utilized to develop most of the content within this project. This demonstrates the potential of AI-driven development and the seamless integration of AI-generated content to suit user needs.

---

If you liked the project, give the repo a star.
Feel free to contribute :)

