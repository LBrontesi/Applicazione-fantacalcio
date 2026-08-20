#!/usr/bin/env node
/* Read Guida's Realm database without mutating it. */
const fs = require("fs");
const Realm = require("realm");

const [source, destination] = process.argv.slice(2);
if (!source || !destination) {
  console.error("Usage: guida_realm_reader.js SOURCE.realm OUTPUT.json");
  process.exit(2);
}

try {
  const realm = new Realm({ path: source, readOnly: true });
  const wanted = [
    "PlayerObject",
    "ProbableStartersObject",
    "ProbableSetPiecesTakersObject",
    "ProbableDoubtObject",
  ];
  const dump = Object.fromEntries(wanted.map((name) => {
    const present = realm.schema.some((schema) => schema.name === name);
    return [name, present ? Array.from(realm.objects(name)).map((row) => row.toJSON()) : []];
  }));
  fs.writeFileSync(destination, JSON.stringify(dump));
  realm.close();
  process.exit(0);
} catch (error) {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
}
