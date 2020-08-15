import { TrendingService } from "./services/trending";
import { connection } from "./config/database/mongo";
class Main {
  trendingService: TrendingService;
  constructor() {
    this.trendingService = new TrendingService();
  }
  start() {
    this.trendingService.stream();
  }
}
console.log("-----------------------");
console.log("Date: " + new Date());
connection
  .then(() => {
    console.log("Connected to database");
    new Main().start();
  })
  .catch(() => {
    console.log("Error connecting to database");
  });
