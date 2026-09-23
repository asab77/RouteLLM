-- Public, development-only credentials. Never use this setup for production.
CREATE ROLE routellm LOGIN PASSWORD 'local_dev_only';
ALTER DATABASE routellm OWNER TO routellm;
REVOKE ALL ON DATABASE routellm FROM PUBLIC;
GRANT CONNECT ON DATABASE routellm TO routellm;

CREATE ROLE routellm_test LOGIN PASSWORD 'local_test_only';
CREATE DATABASE routellm_test OWNER routellm_test;
REVOKE ALL ON DATABASE routellm_test FROM PUBLIC;
GRANT CONNECT ON DATABASE routellm_test TO routellm_test;
