import pandas as pd
from dateutil import parser
import numpy as np
import pymysql
from datetime import datetime, timedelta
import re
import toad


def parse_date(value):
    """
    尝试将各种格式的日期字符串转换为Pandas的日期类型。
    """
    if pd.isnull(value):
        return pd.NaT
    if isinstance(value, str):
        try:
            # 尝试使用 dateutil 的 parser 解析日期
            return pd.to_datetime(parser.parse(value, yearfirst=True, fuzzy=True))
        except Exception as e:
            # 解析失败
            raise pd.to_datetime('9999-12-31')
    else:
        try:
            return pd.to_datetime(value)
        except Exception as e:
            # 解析失败
            raise pd.to_datetime('9999-12-31')


class DatabaseOperator:
    def __init__(self, host, username, password, database):
        try:
            self.connection = pymysql.connect(host=host,
                                              user=username,
                                              password=password,
                                              database=database,
                                              charset='utf8mb4',
                                              cursorclass=pymysql.cursors.DictCursor)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.connection = None

    def close_connection(self):
        if self.connection is not None:
            self.connection.close()
            print(self.connection._closed)

    def ping(self):
        try:
            if self.connection is not None:
                return self.connection.ping() is None
            else:
                return False
        except Exception as e:
            print(f"Error occurred: {e}")
            return False

    def get_table_structure(self, table_name):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
                result = cursor.fetchone()
                ddl_statement = result['Create Table']

            columns_info = []
            primary_keys = re.findall(r'PRIMARY KEY \(`(\w+)`\)', ddl_statement, re.IGNORECASE)

            # 修正并细化索引匹配逻辑
            index_matches = re.findall(r'(?:UNIQUE\s+)?KEY `(\w+)` \(([\w`,\s]+)\)', ddl_statement, re.IGNORECASE)
            index_columns = {index_name: [col.strip('` ') for col in cols.split(',')] for index_name, cols in
                             index_matches}

            pattern = r"`(\w+)`\s+(\w+(?:\(\d*,*\d*\))?)(?:\s*\b(?!(?:COMMENT)\b)\w+\b\s*)*(?:COMMENT\s*'(.*)')?"
            matches = re.findall(pattern, ddl_statement, re.IGNORECASE)

            for match in matches:
                field_name, field_type, field_comment = match
                field_name = field_name.strip('`')  # 确保移除字段名周围的反引号
                is_primary_key = field_name in primary_keys
                # 更精确的索引判断逻辑
                is_index = field_name in index_columns.keys() or any(
                    field_name in cols for cols in index_columns.values()) or is_primary_key

                columns_info.append({
                    '字段名': field_name,
                    '字段类型': field_type,
                    '字段comment': field_comment if field_comment else None,
                    '是否主键': '是' if is_primary_key else '否',
                    '是否索引': '是' if is_index else '否'
                })

            df = pd.DataFrame(columns_info)
            return df
        except Exception as e:
            print(f"Error occurred: {e}")
            return pd.DataFrame()

    def get_row_counts(self, table_name):
        """
        获取指定表的总记录数、去重记录数和主键名
        :param table_name: 表名
        :return: 总记录数, 去重记录数， 主键名
        """
        try:
            with self.connection.cursor() as cursor:
                # 查询总记录数
                total_count_query = f"SELECT COUNT(1) FROM `{table_name}`"
                cursor.execute(total_count_query)
                total_count = cursor.fetchone()['COUNT(1)']

                # 去掉自增主键后的列表
                queryColumnsWithoutAutoIncr = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE " \
                                              f"TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s " \
                                              f"AND COLUMN_KEY != 'PRI' AND EXTRA != 'auto_increment'"
                cursor.execute(queryColumnsWithoutAutoIncr, (table_name,))
                columns = cursor.fetchall()
                columnsStr = ", ".join([col["COLUMN_NAME"] for col in columns])

                # 查询去重记录数，通常基于表的所有列进行去重计数
                distinct_count_query = f"SELECT COUNT(1) FROM (SELECT DISTINCT {columnsStr} FROM `{table_name}`) a"
                cursor.execute(distinct_count_query)
                distinct_count = cursor.fetchone()['COUNT(1)']

                primary_column_query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = " \
                                       f"'{table_name}' AND COLUMN_KEY = 'PRI';"
                cursor.execute(primary_column_query)
                primary_column_name = cursor.fetchone()['COLUMN_NAME']

                return total_count, distinct_count, primary_column_name
        except pymysql.MySQLError as e:
            print(f"Error occurred while getting row counts: {e}")
            return None, None, None

    def get_date_time_fields(self, table_name):
        try:
            with self.connection.cursor() as cursor:

                # 查询日期字段
                queryDate = f"SELECT column_name,data_type FROM information_schema.columns WHERE table_schema = " \
                            f"DATABASE() AND " \
                            f"table_name = %s AND data_type IN ('date', 'datetime', 'timestamp')"
                cursor.execute(queryDate, (table_name,))
                datetime_columns = cursor.fetchall()
                # 查询为空时 cursor.fetchall() 返回的是元组(),需要转为List
                datetime_columns = list(datetime_columns)

                date_keywords = ['date', 'time', 'year', 'month', 'day']
                # 查询是否存在varchar类型的日期字段
                queryStrDate = f"SELECT column_name,data_type FROM information_schema.columns WHERE table_schema = " \
                               f"DATABASE() AND " \
                               f"table_name = %s AND data_type = 'varchar'"
                cursor.execute(queryStrDate, (table_name,))
                possible_date_fields = cursor.fetchall()
                datetime_str_columns = [col for col in possible_date_fields if any(keyword in col['COLUMN_NAME'].lower() for keyword in date_keywords)]

                datetime_columns += datetime_str_columns
                details = []

                if datetime_columns:
                    min_max_queries = [
                        f"(SELECT MIN({col['COLUMN_NAME']}) AS `{col['COLUMN_NAME']}_min`, MAX({col['COLUMN_NAME']}) AS `{col['COLUMN_NAME']}_max` FROM {table_name})"
                        for col in datetime_columns
                    ]
                    full_query = " UNION ALL ".join(min_max_queries)

                    with self.connection.cursor() as cursor:
                        cursor.execute(full_query)
                        results = cursor.fetchall()

                        # 封装结果
                        for idx, col_info in enumerate(datetime_columns):
                            col_name, col_type = col_info['COLUMN_NAME'], col_info['DATA_TYPE']
                            min_val, max_val = results[idx][f'{col_name}_min'], results[idx][f'{col_name}_max']
                            details.append({
                                '字段名': col_name,
                                '字段类型': col_type,
                                '最早记录时间': min_val,
                                '最新记录时间': max_val
                            })

                df = pd.DataFrame(details) if details else pd.DataFrame(
                    columns=['字段名', '字段类型', '最早记录时间', '最新记录时间'])
                return df
        except pymysql.MySQLError as e:
            print(f"Error occurred while getting datetime fields: {e}")
            return pd.DataFrame()

    def get_tables(self):
        try:
            with self.connection.cursor() as cursor:
                # 查询总记录数
                show_table = "SHOW TABLES;"
                cursor.execute(show_table)
                results = cursor.fetchall()
                children = []

                for res in results:
                    value = list(res.values())[0]
                    children.append({'title': value, 'key': value})
                return children
        except pymysql.MySQLError as e:
            print(f"Error occurred while getting tables: {e}")
            return []

    def table_detect(self, sql):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                column_types = [desc[1] for desc in cursor.description]
                df = pd.DataFrame(data=results, columns=column_names)
                # 获取所有列的类型，并将DECIMAL类型的列转换为float64
                for i, col_type in enumerate(column_types):
                    if col_type == pymysql.FIELD_TYPE.NEWDECIMAL:
                        df[column_names[i]] = df[column_names[i]].astype('float64')
                df = toad.detect(df)
                return df

        except pymysql.MySQLError as e:
            print(f"Error occurred while getting tables: {e}")
            return []

    def distribution_detect(self, sql: str, date_column: str, stat_type: str):
        try:
            # 创建数据库连接
            with self.connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(data=results, columns=column_names)

            # 尝试将日期列转换为标准日期格式yyyy-MM-dd
            try:
                df[date_column] = df[date_column].apply(parse_date)
            except Exception as e:
                raise ValueError(f"无法将日期列转换为标准日期格式: {e}")

            # 新增列
            df['year'] = df[date_column].dt.year
            df['year_month'] = df[date_column].dt.to_period('M')
            df['year_quarter'] = df[date_column].dt.to_period('Q')

            # 获取所有列的类型，并将DECIMAL类型的列转换为float64
            column_names = [desc[0] for desc in cursor.description]
            column_types = [desc[1] for desc in cursor.description]
            for i, col_type in enumerate(column_types):
                if col_type == pymysql.FIELD_TYPE.NEWDECIMAL:
                    df[column_names[i]] = df[column_names[i]].astype('float64')

            # 选择统计类型
            if stat_type == 'monthly':
                period_col = 'year_month'
                freq = 'M'
                periods = 12
            elif stat_type == 'quarterly':
                period_col = 'year_quarter'
                freq = 'Q'
                periods = 4
            elif stat_type == 'yearly':
                period_col = 'year'
                freq = 'A'
                periods = 1
            else:
                raise ValueError("stat_type 参数必须是 'monthly', 'quarterly' 或 'yearly'")

            # 获取当前日期
            today = datetime.today()

            # 统计非日期字段
            result = pd.DataFrame()
            non_date_columns = [col for col in df.columns if
                                col not in [date_column, 'year', 'year_month', 'year_quarter']]

            for column in non_date_columns:
                # 按指定统计类型计数
                count = df.groupby(period_col)[column].count().rename(f'{column}_计数')

                # 确保所有期都有数据，缺失期用0填充
                start_date = df[date_column].min()
                end_date = df[date_column].max()
                all_periods = pd.period_range(start=start_date, end=end_date, freq=freq)
                count = count.reindex(all_periods, fill_value=0)

                # 计算同比增长 (Year-Over-Year, YoY, yoy_growth)
                yoy_growth = count.pct_change(periods=periods).rename(f'{column}_同比增长').mul(100).apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) and not np.isinf(x) else None)

                # 计算环比增长 (Month-Over-Month, MoM; Quarter-Over-Quarter, QoQ; Year-Over-Year, YoY)
                mom_growth = count.pct_change(periods=1).rename(f'{column}_环比增长').mul(100).apply(
                    lambda x: f"{x:.2f}%" if pd.notna(x) and not np.isinf(x) else None)

                # 计算近一年计数和近三年计数
                recent_1_year_counts = []
                recent_3_years_counts = []
                for index in count.index:
                    period_start = index
                    if period_start.freqstr == 'M':
                        one_year_ago_period = period_start - 12
                        three_years_ago_period = period_start - 36
                    elif period_start.freqstr == 'Q':
                        one_year_ago_period = period_start - 4
                        three_years_ago_period = period_start - 12
                    elif period_start.freqstr == 'A':
                        one_year_ago_period = period_start - 1
                        three_years_ago_period = period_start - 3

                    recent_1_year_count = df[df[date_column].apply(
                        lambda x: one_year_ago_period <= x.to_period(freq) <= period_start)][
                        column].count()
                    recent_3_years_count = df[df[date_column].apply(
                        lambda x: three_years_ago_period <= x.to_period(freq) <= period_start)][
                        column].count()

                    recent_1_year_counts.append(recent_1_year_count)
                    recent_3_years_counts.append(recent_3_years_count)

                # 若该列为数值，计算平均值
                if df[column].dtype in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
                    mean = df.groupby(period_col)[column].mean().round(2).rename(f'{column}_平均值')
                    # 合并统计结果
                    stats = pd.concat([count, mean, yoy_growth, mom_growth], axis=1)
                else:
                    stats = pd.concat([count, yoy_growth, mom_growth], axis=1)
                stats[f'{column}_最近一年计数'] = recent_1_year_counts
                stats[f'{column}_最近三年计数'] = recent_3_years_counts

                # 合并结果到总的DataFrame
                if result.empty:
                    result = stats
                else:
                    result = result.join(stats, how='outer')

            # 按时间倒叙排序
            result = result.sort_index(ascending=False)

            return result
            # return df

        except pymysql.MySQLError as e:
            raise Exception(f"SQL执行错误: {e}")

    def get_table_columns(self, table_name):
        try:
            with self.connection.cursor() as cursor:
                # 准备SQL查询语句，获取指定表的字段信息
                sql = """
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s;
                    """
                # 填充表所在的数据库名和表名
                cursor.execute(sql, (self.connection.db, table_name))
                # 获取所有字段名
                columns = [col['COLUMN_NAME'] for col in cursor.fetchall()]

            return columns

        except pymysql.MySQLError as e:
            print(f"查询时发生错误: {e}")
            return []

    def import_from_csv(self, file_name, encoding, table_name):
        try:
            data = pd.read_csv(file_name, encoding=encoding)
            with self.connection.cursor() as cursor:
                # DataFrame的第一行是列名，且数据库表结构与DataFrame列匹配
                # 将DataFrame转换为SQL插入语句列表
                rows = data.itertuples(index=False, name=None)
                insert_query = "INSERT INTO {} ({}) VALUES ({})".format(
                    table_name,
                    ', '.join(data.columns),
                    ', '.join(['%s'] * len(data.columns))
                )

                current_chunk = []
                chunk_size = 1000

                for row in rows:
                    current_chunk.append(row)
                    if len(current_chunk) == chunk_size:
                        cursor.executemany(insert_query, current_chunk)
                        self.connection.commit()  # 提交当前批次的事务
                        current_chunk = []  # 重置当前批次的数据列表

                # 处理最后一批数据
                if current_chunk:
                    cursor.executemany(insert_query, current_chunk)
                    self.connection.commit()  # 确保提交最后一部分数据
        except pymysql.MySQLError as e:
            print(f"查询时发生错误: {e}")

