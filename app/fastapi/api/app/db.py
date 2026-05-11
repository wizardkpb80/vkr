import sys
import pg8000
from dbutils.pooled_db import PooledDB
from .config import DB_CONNECTION_PARAMS
from .logging import logger

connection_pool = None


def init_db_pool(minconn=5, maxconn=20):
    global connection_pool
    try:
        logger.info(f"Initializing connection pool with minconn={minconn} and maxconn={maxconn}")
        connection_pool = PooledDB(
            creator=pg8000,
            mincached=minconn,
            maxcached=maxconn,
            maxconnections=maxconn,
            blocking=True,
            **DB_CONNECTION_PARAMS
        )
        logger.info("Connection pool created successfully.")
    except Exception as error:
        logger.error(f"Error while initializing the connection pool: {error}")
        sys.exit(1)


def connect():
    init_db_pool(minconn=5, maxconn=20)
    if connection_pool:
        try:
            logger.info('DB : Getting connection from the pool.')
            conn = connection_pool.getconn()
            logger.debug("DB : Connection successfully obtained from the pool!")
            return conn
        except Exception as error:
            logger.error(f"DB : Error while getting connection from pool: {error}")
            sys.exit(1)
    else:
        logger.error("DB : Connection pool is not initialized.")
        sys.exit(1)


def return_connection(conn):
    if connection_pool:
        try:
            connection_pool.putconn(conn)
            logger.debug("DB : Connection returned to the pool.")
        except Exception as error:
            logger.error(f"DB : Error while returning connection to pool: {error}")
    else:
        logger.error("DB : Connection pool is not initialized.")


def close_db_pool():
    global connection_pool
    if connection_pool:
        try:
            connection_pool.closeall()
            logger.info("DB : All connections in the pool have been closed.")
        except Exception as error:
            logger.error(f"DB : Error while closing the connection pool: {error}")