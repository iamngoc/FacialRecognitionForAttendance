-- Sleepy Durian World – Database schema
-- it will be automatically executed at the first docker-start

-- pgvector: allows 512-D Embeddings as column typ
CREATE EXTENSION IF NOT EXISTS vector;

-- TABLE 1: employees  (Database A)
-- Store employees of Company and  + AI fingerprint

CREATE TABLE IF NOT EXISTS employees (
    id  SERIAL  PRIMARY KEY,
    name    VARCHAR(100)       NOT NULL,
    email   VARCHAR(200)      NOT NULL UNIQUE,
    department  VARCHAR(100)     DEFAULT 'IT',
    position_   VARCHAR(100)      DEFAULT 'UA',
    date_of_birth   DATE,
    entry_date  DATE DEFAULT     CURRENT_DATE,

    -- ArcFace Embedding: 512 numbers = facial fingerprint
    -- two photos of the same person -> cos similarity ~0.8-0.99
    embedding   vector(512),
    photo_path  VARCHAR(500),
    active   BOOLEAN    DEFAULT TRUE,
    created_in  DATE    DEFAULT NOW(),
    updated_in  DATE    DEFAULT NOW()
);

-- TABLE 1: timerecording  (Database B)
-- each recognized scan -> new entry

CREATE TABLE IF NOT EXISTS timerecording (
    id  SERIAL   PRIMARY KEY,
    employee_id INTEGER     NOT NULL
                REFERENCES employees(id) ON DELETE CASCADE,
    scan_time   TIMESTAMP   NOT NULL DEFAULT,
    scan_date   DATE    NOT NULL DEFAULT CURRENT_DATE,
    scan_type   VARCHAR(20)     NOT NULL
                CHECK (scan_type IN ('COME', 'GO')),
    camera_id   VARCHAR(100)    DEFAULT 'entrance_00',
    confidence_score    FLOAT   NOT NULL
                        CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    snapshot_path   VARCHAR(500)
);

-- quick queries by date/employee
CREATE INDEX IF NOT EXISTS idx_by_employee
    ON timerecording(employee_id, scan_date);
CREATE INDEX IF NOT EXISTS idx_by_date
    ON timerecording(scan_date DESC)

-- View: office hours summary
CREATE OR REPLACE VIEW office_hours AS
SELECT
    e.id        AS employee_id,
    e.name,
    e.department,
    t.scan_date     AS date_,
    MIN(CASE WHEN t.scan_type = 'COME' THEN t.scan_time END) AS arrival,
    MAX(CASE WHEN t.scan_type ) 'GO' THEN t.scan_time END) AS exit,
    ROUND(
        EXTRACT(EPOCH FROM (
            MAX(CASE WHEN t.scan_type='GO' THEN t.scan_time END) -
            MIN(CASE WHEN t.scan_type='COME' THEN t.scan_time END)
            )) /3600.0, 2)  AS working_hours
FROM employee e
JOIN timerecording t on e.id = t.employee_id
GROUP BY e.id, e.name, e.department, t.scan_date
ORDER BY z.scan_date DESC, e.name;

-- original data: Sleepy Durian World Employees
INSERT INTO employee (name, email, department, position_, date_of_birth, entry_date)

VALUES
    ('Max Schmitt', 'schmitt@sdw.de', 'IT', 'Senior', '2001-03-01', '2025-05-01'),
    ('Anna Nguyen', 'nguyen@sdw.de', 'IT', 'Junior', '2000-12-3', '2025-12-01')
   ON CONFLICT (email) DO NOTHING;

DO $$ BEGIN
RAISE NOTICE '************************'
RAISE NOTICE 'Sleepy Durian World initialized';
RAISE NOTICE 'Tables: employees, timerecording';
RAISE NOTICE 'View: office_hours';
RAISE NOTICE 'Entries: Max Schmitt, Anna Nguyen';
END $$
