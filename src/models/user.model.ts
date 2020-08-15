import { Schema, model } from "mongoose";
import { IUser, IUserModel } from "../interfaces/schemas/users.interfaces";

const UserSchema = new Schema<IUser>(
  {
    userId: { type: String, required: true },
    name: { type: String, required: true },
    username: { type: String, required: true },
    location: { type: String, required: false },
    description: { type: String, required: false },
    verified: { type: Boolean, required: true },
    followers: { type: Number, required: true },
    friends: { type: Number, required: true },
    profileUrl: { type: String, required: false },
    backgroundUrl: { type: String, required: false },
    favourites: { type: Number, required: true },
    statuses: { type: Number, required: true },
    url: { type: String, required: false },
    counter: { type: Number, default: 0 },
    createdAt: { type: String, required: true },
  },
  { id: false, versionKey: false }
);

export const userModel = model<IUserModel>("users", UserSchema);
