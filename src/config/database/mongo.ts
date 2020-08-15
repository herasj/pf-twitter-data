import { connect } from "mongoose";
import { environment } from "../environment/environment";

const options = {
  useNewUrlParser: true,
  useFindAndModify: false,
  useCreateIndex: true,
  useUnifiedTopology: true,
};

export const connection = connect(environment.mongoUri, options);
