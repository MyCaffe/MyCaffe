﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Data.SqlClient;
using System.Data.EntityClient;
using MyCaffe.db.image;

namespace MyCaffe.test.automated
{
    public class TestEntitiesConnection
    {
        public TestEntitiesConnection()
        {
        }

        public static string CreateConnectionString(string strInstance)
        {
            string strProviderName = "System.Data.SqlClient";
            string strServerName = strInstance;
            string strDatabaseName = "Testing";
            SqlConnectionStringBuilder sqlBuilder = new SqlConnectionStringBuilder();
            EntityConnectionStringBuilder builder = new EntityConnectionStringBuilder();

            sqlBuilder.DataSource = strServerName;
            sqlBuilder.InitialCatalog = strDatabaseName;
            sqlBuilder.IntegratedSecurity = true;

            string strProviderString = sqlBuilder.ToString();

            builder.Provider = strProviderName;
            builder.ProviderConnectionString = strProviderString;
            builder.Metadata = @"res://*/TestingModel.csdl|
                                 res://*/TestingModel.ssdl|
                                 res://*/TestingModel.msl";

            return builder.ToString();
        }

        public static TestingEntities CreateEntities(string strInstance = null)
        {
            if (strInstance == null)
                strInstance = EntitiesConnection.GlobalDatabaseConnectInfo.Server;

            return new TestingEntities(CreateConnectionString(strInstance));
        }
    }
}
