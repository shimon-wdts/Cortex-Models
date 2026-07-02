CREATE TABLE IF NOT EXISTS t_cage_chip_tray_scan (
  table_id integer,
  total_tray_value numeric,
  scan_dtm timestamptz,
  scan_type text,
  table_name text,
  pit_name text,
  gaming_area text,
  top_tray_chip_count integer,
  bottom_tray_chip_count integer
);

CREATE TABLE IF NOT EXISTS t_chip (
  topology_id integer,
  chip_id text,
  denom numeric,
  updated_dtm timestamptz,
  owner_type text,
  status text,
  last_transaction text
);

CREATE TABLE IF NOT EXISTS t_chip_txn_update (
  created_dtm timestamptz,
  txn_update_type text,
  txn_update_value numeric,
  from_topology_id integer,
  to_topology_id integer,
  chips text
);

CREATE TABLE IF NOT EXISTS t_bet (
  table_id integer,
  payout_complete_dtm timestamptz,
  casino_win numeric,
  casino_loss_from_nn numeric
);

CREATE TABLE IF NOT EXISTS t_topology (
  topology_id integer,
  name text,
  description text,
  status text,
  device_type text,
  game_type text,
  table_type text
);

TRUNCATE TABLE
  t_cage_chip_tray_scan,
  t_chip,
  t_chip_txn_update,
  t_bet,
  t_topology;

INSERT INTO t_cage_chip_tray_scan
  (table_id, total_tray_value, scan_dtm, scan_type, table_name, pit_name, gaming_area, top_tray_chip_count, bottom_tray_chip_count)
VALUES
  (64, 1000, '2026-05-10 11:55:00+00', 'AUTO', 'Demo Table 64', 'Pit A', 'Main Floor', 12, 8),
  (66, 2200, '2026-05-10 11:54:00+00', 'AUTO', 'Demo Table 66', 'Pit A', 'Main Floor', 16, 12),
  (72, 700, '2026-05-10 11:56:00+00', 'AUTO', 'Demo Table 72', 'Pit B', 'Main Floor', 10, 6);

INSERT INTO t_chip
  (topology_id, chip_id, denom, updated_dtm, owner_type, status, last_transaction)
SELECT 64, '64-' || gs::text, 100, '2026-05-10 11:50:00+00', 'TABLE', 'ACTIVE', 'DEMO'
FROM generate_series(1, 14) AS gs;

INSERT INTO t_chip
  (topology_id, chip_id, denom, updated_dtm, owner_type, status, last_transaction)
SELECT 66, '66-' || gs::text, 100, '2026-05-10 11:50:00+00', 'TABLE', 'ACTIVE', 'DEMO'
FROM generate_series(1, 20) AS gs;

INSERT INTO t_chip
  (topology_id, chip_id, denom, updated_dtm, owner_type, status, last_transaction)
SELECT 72, '72-' || gs::text, 100, '2026-05-10 11:50:00+00', 'TABLE', 'ACTIVE', 'DEMO'
FROM generate_series(1, 10) AS gs;

INSERT INTO t_chip_txn_update
  (created_dtm, txn_update_type, txn_update_value, from_topology_id, to_topology_id, chips)
VALUES
  ('2026-05-10 11:25:00+00', 'CHIPS_OUT', 4800, 64, NULL, '["64-1","64-2","64-3","64-4"]'),
  ('2026-05-10 11:30:00+00', 'CHIPS_OUT', 3600, 66, NULL, '["66-1","66-2","66-3"]'),
  ('2026-05-10 11:35:00+00', 'CHIPS_OUT', 5200, 72, NULL, '["72-1","72-2","72-3","72-4"]'),
  ('2026-05-10 11:45:00+00', 'CHIPS_IN', 600, NULL, 66, '["66-9"]');

INSERT INTO t_bet
  (table_id, payout_complete_dtm, casino_win, casino_loss_from_nn)
VALUES
  (64, '2026-05-10 11:40:00+00', 0, 3600),
  (66, '2026-05-10 11:42:00+00', 0, 2800),
  (72, '2026-05-10 11:44:00+00', 0, 4200);

INSERT INTO t_topology
  (topology_id, name, description, status, device_type, game_type, table_type)
VALUES
  (64, 'Demo Table 64', 'Local predicted-fills demo table', 'ACTIVE', 'TABLE', 'BACCARAT', 'LIVE'),
  (66, 'Demo Table 66', 'Local predicted-fills demo table', 'ACTIVE', 'TABLE', 'BACCARAT', 'LIVE'),
  (72, 'Demo Table 72', 'Local predicted-fills demo table', 'ACTIVE', 'TABLE', 'BLACKJACK', 'LIVE');
