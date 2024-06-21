from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from database_operator import DatabaseOperator
import uvicorn
import pandas as pd
import os
import sys

root_path = os.getcwd()
sys.path.append(root_path)

app = FastAPI()


@app.get("/tables")
async def get_tables(host, username, password, database):
    tables = []
    # 尝试建立连接
    operator = DatabaseOperator(host, username, password, database)
    # 判断是否建立成功
    is_connected = operator.ping()
    if is_connected:
        tables = operator.get_tables()
        # 关闭连接
        operator.close_connection()
    else:
        # 关闭连接
        operator.close_connection()
        raise HTTPException(status_code=500, detail="Failed to establish a connection to the database")
    return tables


@app.get("/table_columns")
async def get_table_columns(host, username, password, database, table_name):
    table_columns = []
    # 尝试建立连接
    operator = DatabaseOperator(host, username, password, database)
    # 判断是否建立成功
    is_connected = operator.ping()
    if is_connected:
        table_columns = operator.get_table_columns(table_name)
        # 关闭连接
        operator.close_connection()
    else:
        # 关闭连接
        operator.close_connection()
        raise HTTPException(status_code=500, detail="Failed to establish a connection to the database")
    return table_columns


@app.get("/table_summary")
async def get_table_summary(host, username, password, database, table_name):
    result = {}
    # 尝试建立连接
    operator = DatabaseOperator(host, username, password, database)
    # 判断是否建立成功
    is_connected = operator.ping()
    if is_connected:
        total_count, distinct_count, primary_column_name = operator.get_row_counts(table_name)
        table_structure = operator.get_table_structure(table_name)
        time_fields = operator.get_date_time_fields(table_name)
        if not table_structure.empty:
            table_structure.rename(
                columns={'字段名': 'column', '字段类型': 'type', '字段comment': 'comment', '是否主键': 'isPrimary',
                         '是否索引': 'isIndex'}, inplace=True)
        if not time_fields.empty:
            time_fields.rename(
                columns={'字段名': 'column', '字段类型': 'type', '最早记录时间': 'earliestRecordTime',
                         '最新记录时间': 'lastRecordTime'}, inplace=True)

        table_structure['key'] = table_structure.index
        time_fields['key'] = time_fields.index
        structure_dict = table_structure.to_dict(orient='records')
        time_dict = time_fields.to_dict(orient='records')
        # 关闭连接
        operator.close_connection()
        result = {
            "primary_key": primary_column_name,
            "record_count": total_count,
            "distinct_count": distinct_count,
            "time_columns": time_dict,
            "ddl_columns": structure_dict,
        }
    else:
        # 关闭连接
        operator.close_connection()
        raise HTTPException(status_code=500, detail="Failed to establish a connection to the database")
    return result


@app.get("/table_detect")
async def get_table_detect(host, username, password, database, SQL):
    result = {}
    # 尝试建立连接
    operator = DatabaseOperator(host, username, password, database)
    # 判断是否建立成功
    is_connected = operator.ping()
    if is_connected:
        detect_result = operator.table_detect(SQL)

        detect_result.insert(0, 'key', detect_result.index)
        detect_result['key'] = detect_result['key'].apply(str)
        # 把type类型转为对应字符串
        detect_result['type'] = detect_result['type'].apply(str)
        result = detect_result.to_dict(orient='records')
        for item in result:
            for key, value in item.items():
                if pd.isna(value):
                    item[key] = 'nan'
        # 关闭连接
        operator.close_connection()
    else:
        # 关闭连接
        operator.close_connection()
        raise HTTPException(status_code=500, detail="Failed to establish a connection to the database")
    return result


@app.get("/distribution_detect")
async def get_distribution_detect(host, username, password, database, SQL, dateField):
    try:
        # 尝试建立连接
        operator = DatabaseOperator(host, username, password, database)
        # 判断是否建立成功
        is_connected = operator.ping()
        if is_connected:
            # 先默认按月
            detect_result = operator.distribution_detect(SQL, dateField, "monthly")

            detect_result.insert(0, 'key', detect_result.index)
            detect_result['key'] = detect_result['key'].apply(str)
            detect_dict = detect_result.to_dict(orient='records')
            for item in detect_dict:
                for key, value in item.items():
                    if pd.isna(value):
                        item[key] = 'nan'

            # 获取columns
            columns = detect_result.columns

            # 遍历获取columns，构建所需的结构
            detect_columns = []
            for col in columns:
                if col == 'key':
                    detect_columns.insert(0, {'title': '时间', 'dataIndex': col, 'fixed': 'left', 'width': 150})
                else:
                    detect_columns.append({'title': col, 'dataIndex': col})

            result = {
                "detect_dict": detect_dict,
                "detect_columns": detect_columns,
            }
            # 关闭连接
            operator.close_connection()
        else:
            # 关闭连接
            operator.close_connection()
            return JSONResponse(status_code=400, content={"message": "访问数据库失败，请检查数据库配置是否有误！"})
    except Exception as e:
        error_message = str(e)
        return JSONResponse(status_code=400, content={"message": error_message})

    return result

async def stop_uvicorn():
    uvicorn.server.lifespan.Lifespan("app").shutdown()

if __name__ == '__main__':
    log_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "handlers": {
            "file_handler": {
                "class": "logging.FileHandler",
                "filename": "logfile.log",
            },
        },
        "root": {
            "handlers": ["file_handler"],
            "level": "INFO",
        },
    }
    uvicorn.run("app:app", port=8000, host="localhost", workers=1, reload=False)
    # 打包时使用这条
    # uvicorn.run(app, port=8000, host="localhost", workers=1, reload=False, log_config=log_config)
