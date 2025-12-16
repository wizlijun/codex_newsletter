#!/bin/bash

# Codex Newsletter 自动部署脚本
# 用于创建虚拟环境并安装项目依赖

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Codex Newsletter 自动部署${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python 3 是否安装
echo -e "${YELLOW}[1/5] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3 命令${NC}"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ 找到 Python $PYTHON_VERSION${NC}"
echo ""

# 创建虚拟环境
echo -e "${YELLOW}[2/5] 创建虚拟环境...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}警告: venv 目录已存在${NC}"
    read -p "是否删除现有虚拟环境并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有虚拟环境..."
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ 虚拟环境重新创建完成${NC}"
    else
        echo -e "${GREEN}✓ 使用现有虚拟环境${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
fi
echo ""

# 激活虚拟环境
echo -e "${YELLOW}[3/5] 激活虚拟环境...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
echo ""

# 升级 pip
echo -e "${YELLOW}[4/5] 升级 pip...${NC}"
python -m pip install --upgrade pip -q
echo -e "${GREEN}✓ pip 已升级到最新版本${NC}"
echo ""

# 安装依赖
echo -e "${YELLOW}[5/5] 安装项目依赖...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo -e "${RED}错误: 未找到 requirements.txt 文件${NC}"
    exit 1
fi
echo ""

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}注意: 未找到 .env 文件${NC}"
    echo "请根据 .env.example 创建 .env 文件并配置相关信息"
    echo ""
fi

# 显示成功信息
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  部署完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "下一步操作:"
echo "  1. 运行以下命令激活虚拟环境:"
echo -e "     ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "  2. 配置 .env 文件（如果还未配置）"
echo ""
echo "  3. 运行项目:"
echo -e "     ${YELLOW}python getmail.py${NC}"
echo ""
